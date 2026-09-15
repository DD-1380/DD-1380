import asyncio
import os
from pathlib import Path

import cv2
import numpy as np
from docscan.doc import scan
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from image_transform import warp_to_source
from overlay import overlay
from extract_fields import extract_fields
from checkbox_crop import classify_checkbox, is_checkbox_field

HERE = Path(__file__).parent
os.environ.setdefault("U2NET_HOME", str(HERE / ".cache" / "u2net"))

model = None

# gets the OCR model.
def get_model():
    global model
    if model is None:
        model = ocr_predictor(pretrained=True) # .cuda() remove .cuda() to use CPU. TODO This should be a flag
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
) -> tuple[np.ndarray, np.ndarray, dict]:
    scan_bytes = await asyncio.to_thread(scan, raw_bytes)
    scan_img = cv2.imdecode(np.frombuffer(scan_bytes, np.uint8), cv2.IMREAD_COLOR)

    ocr = await asyncio.to_thread(get_model)
    target = await asyncio.to_thread(ocr_page, ocr, scan_bytes)

    flat = warp_to_source(scan_img, source, target)
    checkboxWords, _textBoxes = get_field_locations(source)
    checkboxResults = classify_checkbox(source, flat, checkboxWords)
    fields = await asyncio.to_thread(extract_fields, flat, source)
    return flat, overlay(source, flat, checkboxResults), fields
