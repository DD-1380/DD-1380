import asyncio
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import cv2

# Import your existing pipeline function
from pipeline import process_document

app = FastAPI(title="DD1380 Transcriptions API")

# In-memory job store: { job_uuid: { "status": "working" | "done" | "failed", "result": dict | None, "error": str | None } }
job_store: Dict[str, Dict[str, Any]] = {}

source_store: Dict[str, Any] = {"source": None}

async def run_ocr_background(job_id: str, raw_bytes: bytes, source_dict: dict):
    try:
        # Run your pipeline's async process_document function
        # process_document returns: flat, overlay_img, fields, checkboxResults
        flat, overlay_img, fields, checkboxResults = await process_document(source_dict, raw_bytes)
        
        # Store successful results
        job_store[job_id] = {
            "status": "done",
            "result": {
                "fields": fields,
                "checkboxes": checkboxResults
            },
            "error": None
        }
    except Exception as e:
        # Handle failures gracefully with a meaningful error message
        job_store[job_id] = {
            "status": "failed",
            "result": None,
            "error": str(e)
        }

@app.put("/source")
async def upload_source(source: Dict[str, Any]):
    """Upload the source OCR JSON (the form template). Jobs use the most recent upload."""
    if "dimensions" not in source or "blocks" not in source:
        raise HTTPException(status_code=422, detail="Not a source OCR JSON: missing 'dimensions' or 'blocks'.")
    source_store["source"] = source
    return {"status": "stored"}

@app.post("/jobs/{job_uuid}", status_code=status.HTTP_202_ACCEPTED)
async def create_job(job_uuid: str, request: Request, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    1. BATDOK uploads images with a job UUID they generate.
    2. We reply with 202 Accepted right away.
    """
    source_dict = source_store["source"]
    if source_dict is None:
        raise HTTPException(status_code=409, detail="No source JSON uploaded. PUT it to /source first.")

    # Read the raw image bytes directly from the request body
    raw_bytes = await request.body()
    
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="No image data received.")
    
    # Initialize job status as working
    job_store[job_uuid] = {
        "status": "working",
        "result": None,
        "error": None
    }
    
    # Kick off processing in the background
    background_tasks.add_task(run_ocr_background, job_uuid, raw_bytes, source_dict)
    
    return {"status": "accepted", "job_uuid": job_uuid}

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