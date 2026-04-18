"""
Google Cloud Storage helper for audio file uploads.

Uploaded files are stored under an "uploads/" prefix with a UUID filename to
prevent collisions. The gs:// URI is passed directly to the Speech-to-Text and
Gemini APIs, which both read audio from GCS without routing it through this server.
"""

import uuid

from google.cloud import storage


def upload_to_gcs(
    bucket_name: str,
    file_bytes: bytes,
    original_filename: str,
    content_type: str,
) -> str:
    """
    Upload raw audio bytes to GCS and return the gs:// URI.

    Args:
        bucket_name:       GCS bucket name (without gs:// prefix).
        file_bytes:        Raw audio content.
        original_filename: Used only to extract the file extension.
        content_type:      MIME type stored as blob metadata.

    Returns:
        A GCS URI of the form "gs://bucket/uploads/<uuid>.<ext>"
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Preserve the file extension so downstream APIs can identify the format.
    ext = original_filename.rsplit(".", 1)[-1] if "." in original_filename else "audio"
    blob_name = f"uploads/{uuid.uuid4()}.{ext}"

    blob = bucket.blob(blob_name)
    blob.upload_from_string(file_bytes, content_type=content_type)

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    return gcs_uri
