"""
Meeting Transcriber — FastAPI application.

Routes:
  GET  /         Serve the single-page web UI.
  POST /upload   Accept an audio file, upload it to GCS, and start a
                 transcription job with the chosen engine.
  GET  /status   Poll the status of a transcription job.

Two transcription engines are supported:
  chirp3  — Google Cloud Speech-to-Text V2 with the Chirp 3 model.
            Returns an LRO (Long-Running Operation) name as the job ID.
            Polling calls get_operation() on the Speech API.
  gemini  — Gemini 2.5 Flash via Vertex AI.
            Runs synchronously in a background thread; returns a local
            "gemini-<uuid>" job ID. Polling reads from an in-memory dict.

The /status endpoint distinguishes the two job types by the ID prefix:
  "gemini-..." → Gemini in-memory store
  anything else → Chirp 3 LRO
"""

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.audio import ensure_supported
from app.gemini import get_gemini_status, start_gemini_transcription
from app.speech import get_transcription_status, start_transcription
from app.storage import upload_to_gcs

app = FastAPI(title="Meeting Transcriber")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

GCS_BUCKET = os.environ["GCS_BUCKET"]

# MIME types accepted at the upload endpoint.
# M4A is accepted here and converted to MP3 in audio.py when using Chirp 3.
ALLOWED_CONTENT_TYPES = {
    "audio/mpeg", "audio/mp3",       # MP3
    "audio/wav", "audio/x-wav",      # WAV
    "audio/flac", "audio/x-flac",    # FLAC
    "audio/ogg",                      # OGG
    "audio/webm", "video/webm",       # WebM (some recorders use video/webm)
    "audio/mp4", "audio/x-m4a",      # M4A / AAC
}

VALID_ENGINES = {"chirp3", "gemini"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    language_code: str = Form(...),
    min_speakers: int = Form(1),
    max_speakers: int = Form(10),
    denoise: bool = Form(False),
    engine: str = Form("chirp3"),
):
    """
    Accept an audio upload, store it in GCS, and start a transcription job.

    Returns {"operation_name": "<job_id>"} where job_id is either a Chirp 3
    LRO name or a "gemini-<uuid>" local job ID.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    if min_speakers < 1 or max_speakers > 10 or min_speakers > max_speakers:
        raise HTTPException(status_code=400, detail="Invalid speaker count range")
    if engine not in VALID_ENGINES:
        raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Chirp 3 cannot handle M4A/AAC — convert to MP3 before uploading.
    # Gemini supports M4A natively, so no conversion is needed for that engine.
    if engine == "chirp3":
        file_bytes, actual_content_type = ensure_supported(file_bytes, file.content_type)
    else:
        actual_content_type = file.content_type

    # If format was converted the filename extension must reflect the new type.
    upload_name = (
        "audio.mp3"
        if actual_content_type != file.content_type
        else (file.filename or "audio")
    )
    gcs_uri = upload_to_gcs(GCS_BUCKET, file_bytes, upload_name, actual_content_type)

    if engine == "gemini":
        job_id = start_gemini_transcription(
            gcs_uri, actual_content_type, language_code, min_speakers, max_speakers
        )
    else:
        job_id = start_transcription(gcs_uri, language_code, min_speakers, max_speakers, denoise)

    return {"operation_name": job_id}


@app.get("/status")
async def status(op: str):
    """
    Poll the status of a transcription job.

    The job type is inferred from the ID prefix:
      "gemini-..." → read from in-memory Gemini job store
      anything else → poll the Chirp 3 LRO via the Speech API
    """
    if not op:
        raise HTTPException(status_code=400, detail="Missing op parameter")

    if op.startswith("gemini-"):
        return get_gemini_status(op)
    return get_transcription_status(op)
