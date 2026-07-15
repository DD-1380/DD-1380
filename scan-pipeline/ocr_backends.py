"""Second-pass field OCR backends.

extract_fields.py crops each field out of the aligned image and calls
ocr() on the crop. The backend used for that per-field OCR is selected via
the OCR_BACKEND env var:

  OCR_BACKEND=doctr (default) - local doctr text recognition model.
  OCR_BACKEND=llm             - Gemma 4 vision-language model served over an
                                 OpenAI-compatible endpoint (Ollama or vLLM).
                                 See llm_ocr.py for connection settings.
"""

import os

import numpy as np

doctr_model = None

def get_doctr_model():
    global doctr_model
    if doctr_model is None:
        from doctr.models import ocr_predictor
        doctr_model = ocr_predictor(pretrained=True)
    return doctr_model

# Run second-pass OCR on an already-cropped image region using doctr.
def ocr_doctr(image_crop: np.ndarray) -> str:
    import cv2
    from doctr.io import DocumentFile

    _, buf = cv2.imencode(".jpg", image_crop)
    doc = DocumentFile.from_images(buf.tobytes())
    result = get_doctr_model()(doc)
    words = [
        word.value
        for page in result.pages
        for block in page.blocks
        for line in block.lines
        for word in line.words
    ]
    return " ".join(words).strip()

# Run second-pass OCR on an already-cropped image region using a Gemma 4
# vision-language model served over an OpenAI-compatible endpoint (Ollama or
# vLLM). Imported lazily so the `openai` dependency and a reachable server
# are only required when this backend is selected.
def ocr_llm(image_crop: np.ndarray) -> str:
    import llm_ocr
    return llm_ocr.ocr(image_crop)

BACKENDS = {
    "doctr": ocr_doctr,
    "llm": ocr_llm,
}

# Run second-pass OCR on an already-cropped image region using the backend
# configured by the OCR_BACKEND env var.
def ocr(image_crop: np.ndarray) -> str:
    if image_crop.size == 0:
        return ""
    backend = os.environ.get("OCR_BACKEND", "doctr")
    return BACKENDS[backend](image_crop)
