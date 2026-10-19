import asyncio
import os
import re
from pathlib import Path

import cv2
import numpy as np
from scan_card import scan
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from image_transform import warp_to_source
from overlay import overlay
from extract_fields import extract_fields
from checkbox_crop import classify_checkbox, is_checkbox_field
from scoring import score_card

HERE = Path(__file__).parent
os.environ.setdefault("U2NET_HOME", str(HERE / ".cache" / "u2net"))

model = None

# gets the OCR model
def get_model():
    global model
    if model is None:
        import torch
        model = ocr_predictor(pretrained=True)
        if torch.cuda.is_available():
            model = model.cuda()
    return model

# OCRs the page.
def ocr_page(model, image_bytes: bytes) -> dict:
    doc = DocumentFile.from_images(image_bytes)
    return model(doc).pages[0].export()

def get_field_locations(source: dict):
    checkboxArr = []
    textBoxArr = []
    
    for block in source["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                if is_checkbox_field(word["value"]):
                    checkboxArr.append(word)
                else:
                    textBoxArr.append(word)
                    
    return checkboxArr, textBoxArr

# processes the document.
async def process_document(
    source: dict,
    raw_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    scan_bytes = await asyncio.to_thread(scan, raw_bytes)
    scan_img = cv2.imdecode(np.frombuffer(scan_bytes, np.uint8), cv2.IMREAD_COLOR)

    ocr = await asyncio.to_thread(get_model)
    target = await asyncio.to_thread(ocr_page, ocr, scan_bytes)

    flat = warp_to_source(scan_img, source, target)
    checkboxWords, _textBoxes = get_field_locations(source)
    checkboxResults = classify_checkbox(source, flat, checkboxWords)
    fields = await asyncio.to_thread(extract_fields, flat, source)
    return flat, overlay(source, flat, checkboxResults), fields, checkboxResults

_sources = {}


def get_source(side: str, sources: dict | None = None) -> dict:
    source = (_sources if sources is None else sources).get(side)
    if source is None:
        raise ValueError(f"No {side} source JSON was provided.")
    return source


# "front" or "back" from a photo name like "#38_Back_Case1.HEIC", None if it doesn't say.
def side_of(name: str) -> str | None:
    match = re.search(r"(?<![a-z])(front|back)(?![a-z])", Path(name).stem, re.IGNORECASE)
    return match.group(1).lower() if match else None


# Groups photos into cards by name: #38_Front_Case1 and #38_Back_Case1 -> "#38_Case1".
def pair_photos(paths) -> dict:
    cards = {}
    for path in map(Path, paths):
        side = side_of(path.name)
        if side:
            key = re.sub(r"[_\- ]?(front|back)(?![a-z])", "", path.stem, flags=re.IGNORECASE)
            cards.setdefault(key, {})[side] = path
    return cards


# Runs each side present through process_document and scores the card. A missing side's items are not_scored.
async def process_card(front_bytes: bytes | None = None, back_bytes: bytes | None = None,
                       sources: dict | None = None) -> dict:
    sides, fields, checkboxes = {}, {}, {}
    for side, raw in (("front", front_bytes), ("back", back_bytes)):
        if raw is None:
            continue
        flat, overlaid, side_fields, side_boxes = await process_document(get_source(side, sources), raw)
        sides[side] = {"aligned": flat, "overlay": overlaid, "fields": side_fields, "checkboxes": side_boxes}
        fields.update(side_fields)
        checkboxes.update(side_boxes)
    return {"sides": sides, "fields": fields, "checkboxes": checkboxes, "score": score_card(fields, checkboxes)}