import asyncio
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import cv2
from pathlib import Path
import json

# Import your existing pipeline function
from pipeline import process_document

app = FastAPI(title="DD1380 Transcriptions API")

# In-memory job store: { job_uuid: { "status": "working" | "done" | "failed", "result": dict | None, "error": str | None } }
job_store: Dict[str, Dict[str, Any]] = {}

# Path to your source reference JSON/dictionary used in the pipeline
# Update this path if your template source json is named differently or located elsewhere
HERE = Path(__file__).parent
DEFAULT_SOURCE_PATH = HERE / "front_source.json"  # Adjust if necessary, or pass a default template

def load_default_source() -> dict:
    if DEFAULT_SOURCE_PATH.exists():
        with open(DEFAULT_SOURCE_PATH, "r") as f:
            return json.load(f)
    return {}

async def run_ocr_background(job_id: str, raw_bytes: bytes):
    try:
        # Load your source reference dictionary
        source_dict = load_default_source()
        
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

@app.post("/jobs/{job_uuid}", status_code=status.HTTP_202_ACCEPTED)
async def create_job(job_uuid: str, request: Request, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    1. BATDOK uploads images with a job UUID they generate.
    2. We reply with 202 Accepted right away.
    """
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
    background_tasks.add_task(run_ocr_background, job_uuid, raw_bytes)
    
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