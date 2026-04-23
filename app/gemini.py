"""
Meeting transcription via Gemini 2.5 Flash on Vertex AI.

How it works:
- Gemini is a multimodal model that can process audio files directly.
- We pass a GCS URI to the model using Part.from_uri(), so the audio bytes
  never flow through this server — Gemini reads the file from GCS itself.
- The model is prompted to transcribe the audio and label each speaker turn.
- We parse its plain-text response into structured segments.

Vertex AI vs Google AI Studio:
- This module uses the Vertex AI SDK (google-cloud-aiplatform), which
  authenticates using Application Default Credentials — the same GCP
  service account used for Speech-to-Text and Cloud Storage.
- No separate API key is needed.

Async strategy:
- Gemini's generate_content() is a synchronous (blocking) call.
- FastAPI runs in an async event loop, so blocking calls must run in a
  thread pool to avoid stalling the server.
- We use a ThreadPoolExecutor and an in-memory dict (_jobs) to track results.
- The job ID (prefixed "gemini-") is returned immediately; the frontend polls
  the same /status endpoint used for Chirp 3 LROs.

Note: _jobs is in-process memory. It works for a single Cloud Run instance
(or local dev) but is lost on restart. For multi-instance deployments,
replace _jobs with Firestore or another shared store.
"""

import concurrent.futures
import logging
import os
import re
import uuid

import vertexai
from vertexai.generative_models import GenerativeModel, Part

logger = logging.getLogger(__name__)

PROJECT_ID = os.environ["PROJECT_ID"]
# Gemini is available in many more regions than Chirp 3.
# asia-southeast1 has full model coverage and is the recommended default.
GEMINI_LOCATION = os.environ.get("GEMINI_LOCATION", "asia-southeast1")

# Model identifier for Vertex AI. Check the Vertex AI model garden for the
# latest stable version: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini
MODEL_ID = "gemini-2.5-flash"

# Initialize the Vertex AI SDK once at module load time.
# This sets the default project and region for all subsequent API calls.
vertexai.init(project=PROJECT_ID, location=GEMINI_LOCATION)

# Thread pool for running blocking Gemini calls without blocking the async server.
# max_workers=4 allows up to 4 concurrent transcriptions.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# In-memory job store: job_id -> result dict
_jobs: dict[str, dict] = {}

# Prompt template for structured transcription output.
# The "Speaker N: <text>" format makes the response easy to parse reliably.
_PROMPT_TEMPLATE = (
    "Transcribe this meeting audio recording.\n"
    "Language: {language_code}.\n"
    "Number of speakers: between {min_speakers} and {max_speakers}.\n\n"
    "Rules:\n"
    "- Label each speaker consistently as Speaker 1, Speaker 2, etc.\n"
    "- Output ONE speaker turn per line in this exact format:\n"
    "  Speaker N: <transcribed text>\n"
    "- Do not include timestamps, headers, summaries, or any other text.\n"
    "- Preserve the original language; do not translate."
)


def start_gemini_transcription(
    gcs_uri: str,
    content_type: str,
    language_code: str,
    min_speakers: int,
    max_speakers: int,
) -> str:
    """
    Start an async Gemini transcription job and return a job ID for polling.

    The job runs in a background thread. Call get_gemini_status() with the
    returned job ID to check progress.
    """
    job_id = f"gemini-{uuid.uuid4()}"
    _jobs[job_id] = {"done": False, "segments": None}

    _executor.submit(
        _run_transcription,
        job_id, gcs_uri, content_type, language_code, min_speakers, max_speakers,
    )

    logger.info("Queued Gemini job %s — model=%s file=%s", job_id, MODEL_ID, gcs_uri)
    return job_id


def get_gemini_status(job_id: str) -> dict:
    """
    Return the current state of a Gemini transcription job.

    Possible return values:
      {"done": False, "segments": None}               — still running
      {"done": True, "segments": [...]}               — success
      {"done": True, "segments": None, "error": "..."} — failed
    """
    return _jobs.get(job_id, {"done": False, "segments": None})


def _run_transcription(
    job_id: str,
    gcs_uri: str,
    content_type: str,
    language_code: str,
    min_speakers: int,
    max_speakers: int,
) -> None:
    """Worker function that runs in a background thread."""
    try:
        model = GenerativeModel(MODEL_ID)

        # Part.from_uri() tells Gemini to read the audio directly from GCS.
        # The audio bytes are never downloaded to this server.
        # content_type must match the actual file encoding (e.g. "audio/mpeg").
        audio_part = Part.from_uri(gcs_uri, mime_type=content_type)

        prompt = _PROMPT_TEMPLATE.format(
            language_code=language_code,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        logger.info("Calling %s for job %s", MODEL_ID, job_id)

        # generate_content() accepts a list of Parts (audio, text, images, etc.)
        # followed by the instruction text as a plain string.
        response = model.generate_content([audio_part, prompt])

        segments = _parse_transcript(response.text)
        logger.info("Gemini job %s complete — %d segments parsed", job_id, len(segments))
        _jobs[job_id] = {"done": True, "segments": segments}

    except Exception as exc:
        logger.exception("Gemini job %s failed", job_id)
        _jobs[job_id] = {"done": True, "segments": None, "error": str(exc)}


def _parse_transcript(text: str) -> list[dict]:
    """
    Parse Gemini's plain-text response into a list of speaker segments.

    Expected input format (one turn per line):
        Speaker 1: Good morning, let's get started.
        Speaker 2: Thanks for joining everyone.

    Returns:
        [{"speaker": "Speaker 1", "text": "Good morning, let's get started."},
         {"speaker": "Speaker 2", "text": "Thanks for joining everyone."}]

    Long turns that Gemini wraps across multiple lines are concatenated.
    """
    segments: list[dict] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(Speaker\s+\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            # Flush the previous speaker's accumulated text
            if current_speaker and current_lines:
                segments.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_lines),
                })
            current_speaker = match.group(1).title()  # normalise to "Speaker N"
            first_line = match.group(2).strip()
            current_lines = [first_line] if first_line else []
        elif current_speaker:
            # Continuation of the previous speaker's turn
            current_lines.append(line)

    # Flush the final segment
    if current_speaker and current_lines:
        segments.append({"speaker": current_speaker, "text": " ".join(current_lines)})

    return segments
