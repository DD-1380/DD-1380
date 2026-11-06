import os
from contextlib import contextmanager

import numpy as np

READERS = ("llm", "trocr", "union")

doctr_model = None

def get_doctr_model():
    global doctr_model
    if doctr_model is None:
        from doctr.models import ocr_predictor
        doctr_model = ocr_predictor(pretrained=True)
    return doctr_model

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

def ocr_llm(image_crop: np.ndarray, context: str | None = None) -> str:
    import llm_ocr
    return llm_ocr.ocr(image_crop, context)

def ocr_trocr(image_crop: np.ndarray, context: str | None = None) -> str:
    import trocr_backend
    return trocr_backend.ocr(image_crop, context or "")

BACKENDS = {
    "doctr": ocr_doctr,
    "llm": ocr_llm,
    "trocr": ocr_trocr,
}

def ocr(image_crop: np.ndarray, context: str | None = None) -> str:
    if image_crop.size == 0:
        return ""
    backend = os.environ.get("OCR_BACKEND", "llm")
    if backend in ("llm", "trocr"):
        return BACKENDS[backend](image_crop, context)
    return BACKENDS[backend](image_crop)

@contextmanager
def reader_backend(name: str):
    if name not in READERS:
        raise ValueError(f"Reader must be one of {', '.join(READERS)}.")
    old_backend = os.environ.get("OCR_BACKEND")
    old_model = os.environ.get("OCR_LLM_MODEL")
    os.environ["OCR_BACKEND"] = name
    if name in ("llm", "union"):
        import llm_ocr
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

def workers() -> int:
    backend = os.environ.get("OCR_BACKEND", "llm")
    if backend == "llm":
        return max(1, int(os.environ.get("OCR_LLM_WORKERS", "4")))
    return 1