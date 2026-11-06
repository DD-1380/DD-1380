import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from injury_crop import is_injury_field
from ocr_backends import ocr, workers

# Shown when Gemma and TrOCR do not read the same text.
DISAGREE = "human input"

# Printed labels that sit inside a field's box, stripped from the start of its reading.
PRINTED_LABELS = {
    "field_notes": r"NOTES\s*:?",
    "field_first_responder_name": r"ER\b[\s:.,-]*",
}

# Extract bounding boxes for all text fields from source.json.
def get_field_boxes(source: dict) -> dict:
    boxes = {}
    for block in source["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                value = word["value"]
                if (
                    value.startswith("field_")
                    and not value.startswith("field_image_")
                    and not is_injury_field(value)
                ):
                    boxes[value] = word["geometry"]
    return boxes

# Crop a region from the aligned image.
def crop(image: np.ndarray, box: tuple) -> np.ndarray:
    h, w = image.shape[:2]
    (x0, y0), (x1, y1) = box
    x0 = max(0, int(x0 * w))
    y0 = max(0, int(y0 * h))
    x1 = min(w, int(x1 * w))
    y1 = min(h, int(y1 * h))
    return image[y0:y1, x0:x1]

def clean_field(field_key: str, text: str) -> str:
    if field_key in PRINTED_LABELS:
        text = re.sub(rf"^\s*{PRINTED_LABELS[field_key]}\s*", "", text, flags=re.IGNORECASE)

    # The BloodPressue boxes have a printed "/", so a reading of just "/" means the box is blank.
    if field_key.startswith("field_blood_") and text.replace("/", "").strip() == "":
        text = ""

    # Symbol-only readings like "▼" mean the box is blank. "/" and "-" are kept as line-throughs.
    if text and not any(c.isalnum() or c in "/-" for c in text):
        text = ""
    return text

def read_fields(aligned_image: np.ndarray, source: dict) -> dict:
    boxes = get_field_boxes(source)
    field_keys = list(boxes.keys())
    crops = [crop(aligned_image, boxes[field_key]) for field_key in field_keys]

    with ThreadPoolExecutor(max_workers=workers()) as pool:
        texts = pool.map(ocr, crops, field_keys)

    return {
        field_key: clean_field(field_key, text)
        for field_key, text in zip(field_keys, texts)
    }

def union_fields(aligned_image: np.ndarray, source: dict) -> dict:
    saved = os.environ.get("OCR_BACKEND")
    try:
        os.environ["OCR_BACKEND"] = "llm"
        gemma = read_fields(aligned_image, source)
        os.environ["OCR_BACKEND"] = "trocr"
        trocr = read_fields(aligned_image, source)
    finally:
        if saved is None:
            os.environ.pop("OCR_BACKEND", None)
        else:
            os.environ["OCR_BACKEND"] = saved
    return {
        field_key: text if text == trocr.get(field_key) else DISAGREE
        for field_key, text in gemma.items()
    }

# Takes aligned image and source JSON, returns extracted field dict.
# Output keys are the field_key names from source.json
def extract_fields(aligned_image: np.ndarray, source: dict) -> dict:
    if os.environ.get("OCR_BACKEND", "llm") == "union":
        output = union_fields(aligned_image, source)
    else:
        output = read_fields(aligned_image, source)
    for field_key, text in output.items():
        print(f"{field_key}: '{text}'")
    return output

if __name__ == "__main__":
    import sys
    from pathlib import Path

    source_path = sys.argv[1]
    image_path = sys.argv[2]

    source = json.loads(Path(source_path).read_text())
    image = cv2.imread(image_path)

    result = extract_fields(image, source)
    print(json.dumps(result, indent=2))