"""
Chirp 3 transcription via Google Cloud Speech-to-Text V2 API.

Key API decisions documented here:
- V2 API is required: Chirp 3 is not available in the older V1 API.
- BatchRecognize is used instead of Recognize because meeting recordings are
  typically longer than 60 seconds. BatchRecognize handles files up to 8 hours
  and accepts audio stored in GCS.
- The default recognizer ("recognizers/_") is used so no named recognizer
  resource needs to be created or managed.
- AudioDetectDecodingConfig lets the API auto-detect the audio encoding,
  avoiding the need to specify format per file. Supported formats: FLAC, WAV,
  MP3, OGG, WEBM. M4A/AAC is not supported and must be converted first
  (see audio.py).
- Speaker diarization assigns a speaker_label to each word, enabling
  dialogue-style output.
- Inline output (InlineOutputConfig) returns results in the API response
  rather than writing them to a separate GCS file.

Chirp 3 supported regions (as of 2025):
  asia-southeast1, europe-west2, europe-west3, northamerica-northeast1
"""

import logging
import os

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

logger = logging.getLogger(__name__)

PROJECT_ID = os.environ["PROJECT_ID"]
LOCATION = os.environ.get("LOCATION", "asia-southeast1")

# The Speech-to-Text V2 API uses regional endpoints.
# Each SpeechClient instance maintains a gRPC connection, so we cache one
# per process rather than creating a new one on every request.
_client: SpeechClient | None = None


def _get_client() -> SpeechClient:
    global _client
    if _client is None:
        _client = SpeechClient(
            client_options={"api_endpoint": f"{LOCATION}-speech.googleapis.com"}
        )
    return _client


def start_transcription(
    gcs_uri: str,
    language_code: str,
    min_speakers: int = 1,
    max_speakers: int = 10,
    denoise: bool = False,
) -> str:
    """
    Submit a Chirp 3 BatchRecognize job for a GCS audio file.

    Returns the LRO (Long-Running Operation) name, which callers use to poll
    for completion via get_transcription_status().
    """
    # RecognitionConfig specifies the model and features to apply.
    # AutoDetectDecodingConfig removes the need to specify audio encoding.
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language_code],  # accepts BCP-47 codes, e.g. "en-US"
        model="chirp_3",
        denoiser_config=cloud_speech.DenoiserConfig(
            denoise_audio=denoise,
            snr_threshold=10,  # 0 means no filtering, just denoising
        ),
        features=cloud_speech.RecognitionFeatures(
            # Speaker diarization adds a speaker_label to each word in the
            # response. The model assigns labels like "1", "2", etc. — it does
            # not identify speakers by name.
            diarization_config=cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=min_speakers,
                max_speaker_count=max_speakers,
            ),
        ),
    )

    # BatchRecognizeRequest ties together the recognizer resource, config,
    # list of files to process, and where to write output.
    request = cloud_speech.BatchRecognizeRequest(
        # "recognizers/_" is the default recognizer — no resource creation needed.
        recognizer=f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_",
        config=config,
        files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            # InlineOutputConfig returns results inside the LRO response,
            # which is simpler than writing to a separate GCS output file.
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    # batch_recognize() starts the job and returns immediately with an LRO.
    # The operation.name is a stable string we can use to poll later.
    operation = _get_client().batch_recognize(request=request)
    logger.info("Submitted BatchRecognize LRO: %s", operation.operation.name)
    return operation.operation.name


def get_transcription_status(operation_name: str) -> dict:
    """
    Check the status of a BatchRecognize LRO.

    Returns a dict:
      {"done": False}                              — still processing
      {"done": True, "segments": [...]}            — success
      {"done": True, "segments": None, "error": "..."} — failure
    """
    op = _get_client().get_operation(request={"name": operation_name})
    logger.info("LRO %s — done=%s error_code=%s", operation_name, op.done, op.error.code)

    if not op.done:
        return {"done": False, "segments": None}

    # op.error is a google.rpc.Status; code 0 means OK.
    if op.error.code:
        msg = op.error.message or f"gRPC error code {op.error.code}"
        logger.error("LRO failed: %s", msg)
        return {"done": True, "segments": None, "error": msg}

    # --- Unpack the response ---
    # op.response is a google.protobuf.Any. Unpacking requires a raw protobuf
    # message object (one that exposes DESCRIPTOR). The google-cloud-speech
    # library uses proto-plus wrappers which do NOT expose DESCRIPTOR directly.
    #
    # Workaround: access ._pb on a proto-plus instance to get the underlying
    # raw protobuf, unpack into that, then wrap back into the proto-plus type
    # for convenient attribute access.
    raw_response = cloud_speech.BatchRecognizeResponse()._pb
    op.response.Unpack(raw_response)
    response = cloud_speech.BatchRecognizeResponse.wrap(raw_response)

    # --- Extract words with speaker labels ---
    # response.results is a map from GCS URI → BatchRecognizeFileResult.
    # Each file result contains an inline_result with a transcript, which in
    # turn holds a list of SpeechRecognitionResult segments. Each segment has
    # one or more alternatives; we take the top alternative (index 0).
    # When speaker diarization is active, alternatives[0].words carries a
    # speaker_label per word. We use those to group words into dialogue turns.
    all_words: list[dict] = []
    fallback_text: list[str] = []

    for uri, file_result in response.results.items():
        if file_result.error.code:
            msg = file_result.error.message or f"File error code {file_result.error.code}"
            logger.error("Transcription error for %s: %s", uri, msg)
            return {"done": True, "segments": None, "error": msg}

        for result in file_result.inline_result.transcript.results:
            if not result.alternatives:
                continue
            alt = result.alternatives[0]
            if alt.words:
                # Diarization path: collect every word with its speaker label.
                for word_info in alt.words:
                    all_words.append({
                        "word": word_info.word,
                        "speaker": word_info.speaker_label or "1",
                    })
            elif alt.transcript:
                # Fallback: diarization not available for this result; keep the
                # plain transcript text and present it as a single speaker.
                fallback_text.append(alt.transcript.strip())

    logger.info(
        "Collected %d diarized words, %d fallback chunks",
        len(all_words), len(fallback_text),
    )

    if all_words:
        segments = _group_by_speaker(all_words)
    elif fallback_text:
        segments = [{"speaker": "Speaker 1", "text": " ".join(fallback_text)}]
    else:
        segments = []

    return {"done": True, "segments": segments}


def _group_by_speaker(words: list[dict]) -> list[dict]:
    """
    Collapse a flat word list into dialogue segments.

    Consecutive words from the same speaker are joined into one segment,
    producing output like:
        [{"speaker": "Speaker 1", "text": "Good morning everyone."},
         {"speaker": "Speaker 2", "text": "Thanks for joining."}]
    """
    segments: list[dict] = []
    current_speaker = words[0]["speaker"]
    current_words = [words[0]["word"]]

    for item in words[1:]:
        if item["speaker"] == current_speaker:
            current_words.append(item["word"])
        else:
            segments.append({
                "speaker": f"Speaker {current_speaker}",
                "text": " ".join(current_words),
            })
            current_speaker = item["speaker"]
            current_words = [item["word"]]

    segments.append({
        "speaker": f"Speaker {current_speaker}",
        "text": " ".join(current_words),
    })
    return segments
