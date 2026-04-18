"""
Audio format conversion for Chirp 3 compatibility.

The Speech-to-Text V2 AutoDetectDecodingConfig supports: FLAC, WAV (LINEAR16),
MP3, OGG_OPUS, and WEBM_OPUS. M4A files use the AAC codec inside an MPEG-4
container, which is not on that list. Uploading an M4A to Chirp 3 returns a
generic "internal error" with no further explanation.

This module converts M4A (and audio/mp4) to MP3 via ffmpeg before the file is
uploaded to GCS. All other formats pass through unchanged.

Gemini does support M4A natively, so conversion is skipped for that engine.
ffmpeg must be installed on the host (it is added to the Docker image).
"""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# MIME types that Chirp 3's AutoDetectDecodingConfig cannot handle.
_NEEDS_CONVERSION = {"audio/mp4", "audio/x-m4a", "audio/m4a"}


def ensure_supported(file_bytes: bytes, content_type: str) -> tuple[bytes, str]:
    """
    Return audio bytes in a format Chirp 3 can process.

    If the content type requires conversion, the audio is transcoded to MP3
    using ffmpeg. Otherwise the original bytes are returned unchanged.

    Returns:
        A (bytes, content_type) tuple ready for upload to GCS and submission
        to the Speech-to-Text API.
    """
    if content_type not in _NEEDS_CONVERSION:
        return file_bytes, content_type

    logger.info("Converting %s → audio/mpeg via ffmpeg", content_type)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input.m4a"
        dst = Path(tmp) / "output.mp3"
        src.write_bytes(file_bytes)

        # -y         overwrite output without prompting
        # -i         input file
        # -codec:a   audio codec (libmp3lame = high-quality MP3 encoder)
        # -q:a 2     variable bitrate quality level (2 ≈ 190 kbps, good quality)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-codec:a", "libmp3lame", "-q:a", "2", str(dst)],
            check=True,
            capture_output=True,  # suppress ffmpeg's verbose console output
        )

        return dst.read_bytes(), "audio/mpeg"
