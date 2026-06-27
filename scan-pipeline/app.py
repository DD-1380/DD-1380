import asyncio
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from pipeline import process_document

# converts BGR to RGB.
def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# processes the document asynchronously with process_document()
async def process_async(source_file: str, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = json.loads(Path(source_file).read_text())
    raw_bytes = cv2.imencode(
        ".jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR),
    )[1].tobytes()
    transformed, overlayed = await process_document(source, raw_bytes)
    return bgr_to_rgb(transformed), bgr_to_rgb(overlayed)

# processes the document synchronously with process_async()
def process(source_file: str, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not source_file or raw is None:
        raise gr.Error("Upload a source JSON and a raw photo.")
    return asyncio.run(process_async(source_file, raw))

# creates the gradio interface.
with gr.Blocks(title="Scan Pipeline") as demo:
    gr.Markdown("Upload a source OCR JSON and a raw photo to scan, align, and overlay.")
    with gr.Row():
        source = gr.File(label="Source OCR JSON", file_types=[".json"])
        raw = gr.Image(
            label="Raw photo",
            type="numpy",
            sources=["upload", "webcam"],
        )
    process_btn = gr.Button("Process", variant="primary")
    with gr.Row():
        transformed_out = gr.Image(label="Transformed")
        overlayed_out = gr.Image(label="Overlayed")
    process_btn.click(process, [source, raw], [transformed_out, overlayed_out])

if __name__ == "__main__":
    demo.launch(share=True)
