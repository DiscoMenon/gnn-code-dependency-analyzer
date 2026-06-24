import asyncio
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from backend.api.schemas import AnalyzeRequest, AnalyzeResponse
from backend.api.pipeline import run_analysis
from typing import Dict
import uuid

router = APIRouter()

# In-memory job store — good enough for now
jobs: dict[str, dict] = {}


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Kicks off analysis as a background job.
    Returns a job_id immediately.
    Frontend polls /status/{job_id} until done.
    """
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "running", "result": None}

    background_tasks.add_task(_run_job, job_id, request.github_url)

    return {"job_id": job_id, "status": "running"}


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]


async def _run_job(job_id: str, github_url: str):
    try:
        # Run in thread pool so it doesn't block the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_analysis, github_url)
        jobs[job_id] = {
            "status": "complete",
            "result": result.model_dump()
        }
    except Exception as e:
        jobs[job_id] = {
            "status": "error",
            "result": None,
            "error": str(e)
        }