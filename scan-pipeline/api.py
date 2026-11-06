import asyncio
import time
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status

from ocr_backends import READERS, reader_backend
from pipeline import process_document

app = FastAPI(title="DD1380 Transcriptions API")

# In-memory job store: { job_uuid: { "status": "working" | "done" | "failed", "result": dict | None, "error": str | None } }
job_store: Dict[str, Dict[str, Any]] = {}

source_store: Dict[str, Any] = {"source": None}
blank_store: Dict[str, Any] = {"image": None}
_ocr_lock = asyncio.Lock()

def decode_image(raw_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the image.")
    return image

async def run_ocr_background(job_id: str, raw_bytes: bytes, source_dict: dict,
                             blank: np.ndarray, reader: str):
    try:
        # One side, feature-aligned to the blank form. Readers share process env, so jobs run one at a time.
        async with _ocr_lock:
            with reader_backend(reader):
                _flat, _overlay_img, fields, checkboxResults = await process_document(
                    source_dict, raw_bytes, blank, "features",
                )
        job_store[job_id] = {
            "status": "done",
            "result": {
                "fields": fields,
                "checkboxes": checkboxResults,
            },
            "error": None,
        }
    except Exception as e:
        job_store[job_id] = {
            "status": "failed",
            "result": None,
            "error": str(e),
        }

@app.put("/source")
async def upload_source(source: Dict[str, Any]):
    """Upload the source OCR JSON (the form template). Jobs use the most recent upload."""
    if "dimensions" not in source or "blocks" not in source:
        raise HTTPException(status_code=422, detail="Not a source OCR JSON: missing 'dimensions' or 'blocks'.")
    source_store["source"] = source
    return {"status": "stored"}

@app.put("/blank")
async def upload_blank(request: Request):
    """Upload the blank form for this one side. Jobs use the most recent upload."""
    raw_bytes = await request.body()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="No image data received.")
    try:
        blank_store["image"] = decode_image(raw_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"status": "stored"}

@app.post("/jobs/{job_uuid}", status_code=status.HTTP_202_ACCEPTED)
async def create_job(job_uuid: str, request: Request, reader: str = "llm",
                     background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    One card side. BATDOK uploads the photo with a job UUID they generate.
    reader is llm, trocr, or union. Reply is 202 Accepted.
    """
    if reader not in READERS:
        raise HTTPException(status_code=422, detail=f"Reader must be one of {', '.join(READERS)}.")
    source_dict = source_store["source"]
    blank = blank_store["image"]
    if source_dict is None:
        raise HTTPException(status_code=409, detail="No source JSON uploaded. PUT it to /source first.")
    if blank is None:
        raise HTTPException(status_code=409, detail="No blank form uploaded. PUT it to /blank first.")

    raw_bytes = await request.body()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="No image data received.")

    job_store[job_uuid] = {
        "status": "working",
        "result": None,
        "error": None,
    }
    background_tasks.add_task(run_ocr_background, job_uuid, raw_bytes, source_dict, blank.copy(), reader)
    return {"status": "accepted", "job_uuid": job_uuid, "reader": reader}

@app.get("/jobs/{job_uuid}")
def get_job(job_uuid: str, wait: int = 0):
    """
    BATDOK asks for the result with a GET for the UUID, and a wait parameter.
    Holds the connection open until done or deadline reached.
    """
    deadline = time.time() + min(wait, 30)
    while True:
        job = job_store.get(job_uuid)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found or upload never arrived.")
        if job["status"] != "working":
            return job  # Returns the finished or failed job state (200 OK)
        if time.time() >= deadline:
            return job  # Returns current status ("working") if wait time expired
        
        time.sleep(0.5)