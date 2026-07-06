from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import pillow_heif

ROOT = Path(__file__).resolve().parents[2]
SCAN_PIPELINE_DIR = ROOT / "scan-pipeline"

if str(SCAN_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(SCAN_PIPELINE_DIR))

from overlay import display_label  # noqa: E402
from pipeline import process_document  # noqa: E402

CHECKBOX_DATASET_DIR = ROOT / "dataset" / "checkbox"
CHECKED_DIR = CHECKBOX_DATASET_DIR / "checked"
UNCHECKED_DIR = CHECKBOX_DATASET_DIR / "unchecked"

pillow_heif.register_heif_opener()


def normalize_uploaded_path(uploaded: Any) -> str:
    if uploaded is None:
        raise ValueError("Missing uploaded file.")

    if isinstance(uploaded, str):
        return uploaded

    for attribute in ("path", "name"):
        value = getattr(uploaded, attribute, None)
        if value:
            return value

    if isinstance(uploaded, dict):
        for key in ("path", "name", "orig_name"):
            value = uploaded.get(key)
            if value:
                return value

    return str(uploaded)


def load_source_page(source_file: Any) -> dict:
    source_path = Path(normalize_uploaded_path(source_file))
    return json.loads(source_path.read_text(encoding="utf-8"))


def raw_image_bytes(raw_path: str) -> bytes:
    path = Path(raw_path)
    if path.suffix.lower() not in {".heic", ".heif"}:
        return path.read_bytes()

    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        output = Path(raw_path).with_suffix(".jpg")
        import io

        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def iter_words(page: dict):
    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            yield from line.get("words", [])


def is_checkbox_word(value: str | None) -> bool:
    if not value:
        return False

    normalized = value.strip().lower()
    if normalized.startswith("field_image_"):
        return True

    label = display_label(value).lower()
    normalized_label = re.sub(r"[\s\-]+", "_", label)
    return "checkbox" in normalized_label or "check_box" in normalized_label


def collect_checkbox_words(page: dict) -> list[dict]:
    return [word for word in iter_words(page) if is_checkbox_word(word.get("value"))]


async def align_scan(source: dict, raw_path: str) -> np.ndarray:
    raw_bytes = raw_image_bytes(raw_path)
    transformed, _ = await process_document(source, raw_bytes)
    return transformed


def crop_word(image: np.ndarray, word: dict, padding: int = 4) -> np.ndarray | None:
    height, width = image.shape[:2]
    (x0, y0), (x1, y1) = word["geometry"]

    left = max(0, int(round(x0 * width)) - padding)
    top = max(0, int(round(y0 * height)) - padding)
    right = min(width, int(round(x1 * width)) + padding)
    bottom = min(height, int(round(y1 * height)) + padding)

    if right <= left or bottom <= top:
        return None

    return image[top:bottom, left:right].copy()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "checkbox"


def unique_path(base_path: Path) -> Path:
    candidate = base_path
    counter = 1
    while candidate.exists():
        candidate = base_path.with_name(f"{base_path.stem}_{counter}{base_path.suffix}")
        counter += 1
    return candidate


def save_crop(sample: dict, label: str) -> Path:
    target_dir = CHECKED_DIR if label == "Marked" else UNCHECKED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_stem = slugify(Path(sample["raw_path"]).stem)
    field_stem = slugify(sample["field_label"])
    file_name = f"{raw_stem}__{sample['crop_index']:04d}__{field_stem}.png"
    target_path = unique_path(target_dir / file_name)

    image_bgr = cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(target_path), image_bgr)
    return target_path


async def build_samples(source_file: Any, raw_files: list[Any]) -> tuple[list[dict], str]:
    source = load_source_page(source_file)
    checkbox_words = collect_checkbox_words(source)

    if not checkbox_words:
        return [], "No checkbox-like fields were found in the source JSON."

    samples: list[dict] = []
    for raw_file in raw_files:
        raw_path = normalize_uploaded_path(raw_file)
        aligned = await align_scan(source, raw_path)

        for crop_index, word in enumerate(checkbox_words):
            crop = crop_word(aligned, word)
            if crop is None:
                continue

            samples.append(
                {
                    "image": crop,
                    "field_label": display_label(word.get("value", "checkbox")),
                    "raw_path": raw_path,
                    "crop_index": crop_index,
                }
            )

    if not samples:
        return [], "Checkbox fields were found, but none produced a valid crop."

    summary = f"Prepared {len(samples)} checkbox crops from {len(raw_files)} image(s)."
    return samples, summary


def current_sample_summary(samples: list[dict], index: int, saved_count: int) -> str:
    if not samples:
        return "Upload a source JSON and one or more images, then load crops."

    if index >= len(samples):
        return f"Review complete. Saved {saved_count} crop(s)."

    sample = samples[index]
    return (
        f"Crop {index + 1} of {len(samples)} | "
        f"Saved {saved_count} | {sample['field_label']} | {Path(sample['raw_path']).name}"
    )


def advance_samples(samples: list[dict], index: int, saved_count: int):
    if index >= len(samples):
        return None, "", current_sample_summary(samples, index, saved_count), samples, index, saved_count

    next_index = index + 1

    if next_index < len(samples):
        next_sample = samples[next_index]
        return (
            next_sample["image"],
            next_sample["field_label"],
            current_sample_summary(samples, next_index, saved_count),
            samples,
            next_index,
            saved_count,
        )

    return None, "", current_sample_summary(samples, next_index, saved_count), samples, next_index, saved_count


def load_review_queue(source_file: Any, raw_files: list[Any]):
    if not source_file or not raw_files:
        raise ValueError("Upload a source JSON and at least one image.")

    samples, summary = asyncio.run(build_samples(source_file, raw_files))
    if not samples:
        return [], 0, 0, None, summary, summary, None

    first_sample = samples[0]
    return (
        samples,
        0,
        0,
        first_sample["image"],
        first_sample["field_label"],
        current_sample_summary(samples, 0, 0),
        "Marked",
    )


def save_and_next(label: str, samples: list[dict], index: int, saved_count: int):
    if not samples:
        raise ValueError("Load checkbox crops first.")

    if index >= len(samples):
        return None, "", current_sample_summary(samples, index, saved_count), samples, index, saved_count, label

    save_crop(samples[index], label)
    saved_count += 1
    next_image, next_label, summary, samples, next_index, saved_count = advance_samples(samples, index, saved_count)
    return next_image, next_label, summary, samples, next_index, saved_count, "Marked"
