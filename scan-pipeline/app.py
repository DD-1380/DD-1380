import asyncio
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

import llm_ocr
import trocr_backend
from ocr_backends import READERS, reader_backend
from pipeline import process_card

READER_CHOICES = [
    (llm_ocr.DEFAULT_MODEL, "llm"),
    (trocr_backend.MODEL_NAME, "trocr"),
    ("union", "union"),
]

# converts BGR to RGB.
def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def to_bgr(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def to_bytes(image: np.ndarray | None) -> bytes | None:
    if image is None:
        return None
    # Feature matching needs the uploaded pixels.
    return cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()

# processes one side asynchronously with process_card()
async def process_async(photo: np.ndarray, sources: dict, blanks: dict):
    card = await process_card(to_bytes(photo), None, sources, blanks, "features")
    result = card["sides"]["front"]
    return (
        bgr_to_rgb(result["aligned"]),
        bgr_to_rgb(result["overlay"]),
        json.dumps(card["fields"], indent=2),
        json.dumps(card["checkboxes"], indent=2),
    )

# processes one side synchronously with process_async()
def process(photo: np.ndarray | None, source_file: str | None,
            blank: np.ndarray | None, reader: str):
    if photo is None:
        raise gr.Error("Upload a photo of the card.")
    if not source_file:
        raise gr.Error("Upload the source JSON to go with the photo.")
    if blank is None:
        raise gr.Error("Upload the blank form to align with feature matching.")
    if reader not in READERS:
        raise gr.Error("Choose a Gemma, TrOCR, or union reader.")
    try:
        sources = {"front": json.loads(Path(source_file).read_text())}
    except (OSError, ValueError) as error:
        raise gr.Error(f"Could not read the source JSON: {error}")
    blanks = {"front": to_bgr(blank)}
    with reader_backend(reader):
        return asyncio.run(process_async(photo, sources, blanks))

# creates the gradio interface.
with gr.Blocks(title="Scan Pipeline") as demo:
    gr.Markdown("Upload one side of a card, its source OCR JSON, and the blank form. Feature matching warps the photo onto that form.")
    photo = gr.Image(label="Photo", type="numpy", sources=["upload", "webcam"])
    source = gr.File(label="Source OCR JSON", file_types=[".json"])
    reader = gr.Radio(
        choices=READER_CHOICES,
        value="llm",
        label="Reader",
    )
    blank = gr.Image(label="Blank form", type="numpy", sources=["upload"])
    process_btn = gr.Button("Process", variant="primary")
    with gr.Row():
        transformed = gr.Image(label="Transformed")
        overlay = gr.Image(label="Overlayed")
    with gr.Row():
        fields_out = gr.Code(label="Extracted Fields", language="json")
        checkboxes_out = gr.Code(label="Checkboxes", language="json")

    process_btn.click(
        process,
        [photo, source, blank, reader],
        [transformed, overlay, fields_out, checkboxes_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)