import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

import llm_ocr
import trocr_backend
from pipeline import process_card

# Field readers already wired in the pipeline. DocTR stays on alignment only.
READERS = [
    (llm_ocr.DEFAULT_MODEL, "llm"),
    (trocr_backend.MODEL_NAME, "trocr"),
]


@contextmanager
def reader_backend(name: str):
    old_backend = os.environ.get("OCR_BACKEND")
    old_model = os.environ.get("OCR_LLM_MODEL")
    os.environ["OCR_BACKEND"] = name
    if name == "llm":
        os.environ["OCR_LLM_MODEL"] = llm_ocr.DEFAULT_MODEL
    try:
        yield
    finally:
        if old_backend is None:
            os.environ.pop("OCR_BACKEND", None)
        else:
            os.environ["OCR_BACKEND"] = old_backend
        if old_model is None:
            os.environ.pop("OCR_LLM_MODEL", None)
        else:
            os.environ["OCR_LLM_MODEL"] = old_model

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

def to_bytes(image: np.ndarray | None, lossless: bool = False) -> bytes | None:
    if image is None:
        return None
    # Feature matching needs the uploaded pixels. DocTR keeps the JPEG it had.
    ext = ".png" if lossless else ".jpg"
    return cv2.imencode(ext, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()

# processes one side asynchronously with process_card()
async def process_async(photo: np.ndarray, sources: dict, blanks: dict, align: str):
    lossless = align == "features"
    card = await process_card(to_bytes(photo, lossless), None, sources, blanks, align)
    result = card["sides"]["front"]
    return (
        bgr_to_rgb(result["aligned"]),
        bgr_to_rgb(result["overlay"]),
        json.dumps(card["fields"], indent=2),
        json.dumps(card["checkboxes"], indent=2),
    )

# processes one side synchronously with process_async()
def process(photo: np.ndarray | None, source_file: str | None,
            align: str, blank: np.ndarray | None, reader: str):
    if photo is None:
        raise gr.Error("Upload a photo of the card.")
    if not source_file:
        raise gr.Error("Upload the source JSON to go with the photo.")
    if align == "features" and blank is None:
        raise gr.Error("Upload the blank form to align with feature matching.")
    if reader not in ("llm", "trocr"):
        raise gr.Error("Choose a Gemma or TrOCR reader.")
    try:
        sources = {"front": json.loads(Path(source_file).read_text())}
    except (OSError, ValueError) as error:
        raise gr.Error(f"Could not read the source JSON: {error}")
    blanks = {"front": to_bgr(blank)} if blank is not None else {}
    with reader_backend(reader):
        return asyncio.run(process_async(photo, sources, blanks, align))

# creates the gradio interface.
with gr.Blocks(title="Scan Pipeline") as demo:
    gr.Markdown("Upload one side of a card and its source OCR JSON. Feature matching warps the photo onto the blank form. DocTR finds the card first, then aligns by words.")
    photo = gr.Image(label="Photo", type="numpy", sources=["upload", "webcam"])
    source = gr.File(label="Source OCR JSON", file_types=[".json"])
    align = gr.Radio(
        choices=[("DocTR words", "doctr"), ("Feature match", "features")],
        value="features",
        label="Alignment",
    )
    reader = gr.Radio(
        choices=READERS,
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

    def show_blank(method: str):
        return gr.update(visible=method == "features")

    align.change(show_blank, align, blank)
    process_btn.click(
        process,
        [photo, source, align, blank, reader],
        [transformed, overlay, fields_out, checkboxes_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)