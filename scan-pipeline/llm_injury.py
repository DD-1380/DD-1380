import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from llm_ocr import crop_to_data_url, get_client, get_model

HERE = Path(__file__).resolve().parent

# Empty DD-1380 used as a reference
BLANK_IMAGE = HERE.parent / "templates" / "blank_dd1380.png"

PROMPT = (
    "You are reading the body picture on a scanned DD Form 1380 (TCCC card). "
    "The form's instructions for this section are: "
    "'Mark an X at the site of the injury(ies) on the body picture. For burn injuries, "
    "circle the burn percentage(s) on the figure. If multiple mechanisms of injury and "
    "multiple injuries, draw a line between the mechanism of injury and the anatomical "
    "site of the injury.' "
    "The picture has PRINTED parts that are never marks: the body outline, muscle and joint "
    "lines, the face and head, and printed numbers (1, 4.5, 9, 18). A printed number is only a "
    "burn if someone drew a circle around it by hand. "
    "List every HANDWRITTEN mark you can see. For each one give: "
    "mark = 'X' for an X, 'circled_burn' for a hand-drawn circle around a printed number, "
    "'line' for a hand-drawn line or stroke, 'other' only if it is none of these; "
    "location = the body part it is on, always including the side from the PATIENT'S point "
    "of view when the body part has one (for example 'right thigh', 'left forearm', 'chest'). "
    "If there are no handwritten marks, return an empty list. Do not guess."
)

REFERENCE_PROMPT = (
    " You get TWO images of the same area. IMAGE 1 is from the filled-in card. IMAGE 2 is the "
    "same area on an EMPTY card. Everything that appears in IMAGE 2 is printed on the form. "
    "Only report handwritten marks that are in IMAGE 1 and NOT in IMAGE 2."
)

VIEW_HINT = {
    "front": " This is the FRONT view: the patient's right side is on the viewer's LEFT.",
    "back": " This is the BACK view: the patient's right side is on the viewer's RIGHT.",
}

MARK_TYPES = ["X", "circled_burn", "line", "other"]
MARK_NAMES = {"X": "X", "circled_burn": "circled burn", "line": "line", "other": "unclear mark"}


SCHEMA = {
    "type": "object",
    "properties": {
        "marks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mark": {"type": "string", "enum": MARK_TYPES},
                    "location": {"type": "string"},
                },
                "required": ["mark", "location"],
            },
        }
    },
    "required": ["marks"],
}


# empty card crops are cached so we don't have to align them every time

_blank_cache: dict = {}


def _load_image(path: Path) -> np.ndarray | None:
    if path.suffix.lower() in {".heic", ".heif"}:
        from PIL import Image
        from pillow_heif import register_heif_opener

        register_heif_opener()
        return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return cv2.imread(str(path))


def _align_blank(path: Path, source: dict) -> np.ndarray:
    # A blank photo that isn't the template image: line it up like a filled card.
    from docscan.doc import scan

    from image_transform import warp_to_source
    from pipeline import get_model as get_ocr_model, ocr_page

    raw = path.read_bytes()
    if path.suffix.lower() in {".heic", ".heif"}:
        raw = cv2.imencode(".jpg", _load_image(path))[1].tobytes()
    scan_bytes = scan(raw)
    scan_img = cv2.imdecode(np.frombuffer(scan_bytes, np.uint8), cv2.IMREAD_COLOR)
    return warp_to_source(scan_img, source, ocr_page(get_ocr_model(), scan_bytes))


def blank_crops_for(source: dict, shape: tuple) -> dict[str, np.ndarray]:
    """Body-diagram crops from the empty card, matching the filled card's aligned size.

    Returns {} if there's no empty card, and everything still works without it.
    Saves the crops to scan-pipeline/.cache/blank_crops/ so you can check they look right.
    """
    from injury_crop import crop_injury_fields

    path = Path(os.environ.get("INJURY_BLANK_IMAGE", BLANK_IMAGE))
    if not path.is_file():
        return {}
    key = (str(path), path.stat().st_mtime, json.dumps(source["dimensions"]), shape[:2])
    if key not in _blank_cache:
        image = _load_image(path)
        if image is None:
            print(f"Could not read the empty card at {path}")
            return {}
        h, w = source["dimensions"]
        same_shape = abs((image.shape[1] / image.shape[0]) - (w / h)) / (w / h) < 0.02
        if not same_shape:
            print("Empty card isn't the template image; aligning it (one time only)...")
            image = _align_blank(path, source)
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        crops = crop_injury_fields(image, source)
        out = HERE / ".cache" / "blank_crops"
        out.mkdir(parents=True, exist_ok=True)
        for name, crop in crops.items():
            cv2.imwrite(str(out / f"{name}.png"), crop)
        _blank_cache[key] = crops
    return _blank_cache[key]


# asking Gemma

def prompt_for(context: str | None = None, with_reference: bool = False) -> str:
    text = PROMPT
    if with_reference:
        text += REFERENCE_PROMPT
    name = (context or "").lower()
    for view, hint in VIEW_HINT.items():
        if view in name:
            text += hint
    return text


def workers() -> int:
    return max(1, int(os.environ.get("OCR_LLM_WORKERS", "4")))


def _summarize(marks: list[dict]):
    """A list with one entry per mark, e.g. ['X on right thigh', 'line on left thigh']."""
    if not marks:
        return "unmarked"
    lines = []
    for m in marks:
        where = f" on {m['location']}" if m["location"] else ""
        lines.append(f"{MARK_NAMES[m['mark']]}{where}")
    return lines

def _parse(text: str) -> list[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"not JSON: {text[:80]!r}")
        data = json.loads(text[start : end + 1])
    marks = []
    for m in data.get("marks", []):
        if isinstance(m, dict) and m.get("mark") in MARK_TYPES:
            marks.append({"mark": m["mark"], "location": str(m.get("location", "")).strip()})
    return marks


def _ask(image_url: str, context: str | None, reference_url: str | None) -> str:
    content = [
        {"type": "text", "text": prompt_for(context, reference_url is not None)},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    if reference_url is not None:
        content.append({"type": "image_url", "image_url": {"url": reference_url}})
    kwargs = dict(
        model=get_model(),
        temperature=0,
        max_tokens=int(os.environ.get("INJURY_MAX_TOKENS", "300")),
        messages=[{"role": "user", "content": content}],
        extra_body={"reasoning_effort": os.environ.get("OCR_LLM_REASONING_EFFORT", "none")},
    )
    try:
        response = get_client().chat.completions.create(
            response_format={"type": "json_schema", "json_schema": {"name": "marks", "schema": SCHEMA}},
            **kwargs,
        )
    except Exception:
        
        response = get_client().chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    text = (response.choices[0].message.content or "").strip()
    try:
        return _summarize(_parse(text))
    except (ValueError, json.JSONDecodeError):
        return f"needs review (model said: {text[:60]})"


def classify_injury(image_crop: np.ndarray, context: str | None = None, blank_crop: np.ndarray | None = None) -> str:
    if image_crop.size == 0:
        return "unmarked"
    reference = None
    if blank_crop is not None and blank_crop.size:
        blank_crop = cv2.resize(blank_crop, (image_crop.shape[1], image_crop.shape[0]))
        reference = crop_to_data_url(blank_crop)
    return _ask(crop_to_data_url(image_crop), context, reference)


def classify_injuries(crops: dict[str, np.ndarray], blank_crops: dict[str, np.ndarray] | None = None) -> dict[str, str]:
    get_client()
    blank_crops = blank_crops or {}
    if crops and not blank_crops:
        print("No empty card found, so printed parts may be read as marks.")
    keys = list(crops)
    with ThreadPoolExecutor(max_workers=workers()) as pool:
        labels = list(pool.map(lambda k: classify_injury(crops[k], k, blank_crops.get(k)), keys))

    output = {}
    for field_key, label in zip(keys, labels):
        output[field_key] = label
        print(f"{field_key}: '{label}'")
    return output


if __name__ == "__main__":
    import sys

    image = cv2.imread(sys.argv[1])
    if image is None:
        raise SystemExit(f"Could not read image: {sys.argv[1]}")
    context = sys.argv[2] if len(sys.argv) > 2 else None
    blank = cv2.imread(sys.argv[3]) if len(sys.argv) > 3 else None
    print(classify_injury(image, context, blank))