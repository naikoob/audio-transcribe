# Meeting Transcriber

A modern, full-stack application for generating speaker-labeled transcripts from meeting recordings. It leverages state-of-the-art AI models from Google Cloud to provide highly accurate results with minimal setup.

## ✨ Features

- **Dual Transcription Engines**:
  - **Google Chirp 3**: Optimized for high-accuracy long-form speech recognition via the Speech-to-Text V2 API.
  - **Gemini 2.5 Flash**: Uses multimodal AI via Vertex AI to provide structured transcription with speaker identification.
- **Speaker Diarization**: Automatically identifies and labels different speakers ("Speaker 1", "Speaker 2", etc.).
- **Background Noise Reduction**: Optional denoising configuration to improve recognition in noisy environments.
- **Multilingual Support**: Supports over 20 languages including English, Spanish, French, German, Mandarin, and more.
- **Web UI**: A clean, dark-mode single-page application for uploading files and viewing transcripts in real-time.
- **Auto-Conversion**: Automatically handles audio format conversions (e.g., M4A to MP3) to ensure compatibility with all engines.

## 🏗️ Architecture

- **Backend**: FastAPI (Python 3.13+)
- **Frontend**: Vanilla JS + CSS (Responsive, Dark Mode)
- **Cloud Infrastructure**:
  - **Google Cloud Storage (GCS)**: Stores uploaded audio for processing.
  - **Speech-to-Text V2**: Processes Chirp 3 transcription jobs.
  - **Vertex AI**: Powers the Gemini transcription engine.
  - **Cloud Run**: Recommended for containerized deployment.

## 🚀 Getting Started

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed.
- A Google Cloud Project with billing enabled.
- Required APIs enabled:
  ```bash
  gcloud services enable speech.googleapis.com storage.googleapis.com aiplatform.googleapis.com
  ```

### Local Setup

1. **Clone the repository** and navigate to the project directory.
2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Authenticate** with Google Cloud:
   ```bash
   gcloud auth application-default login
   ```
4. **Set environment variables**:
   ```bash
   export PROJECT_ID="your-project-id"
   export GCS_BUCKET="your-bucket-name"
   export LOCATION="asia-southeast1" # Region for STT V2
   ```
5. **Run the server**:
   ```bash
   uvicorn app.main:app --reload
   ```

## 🚢 Deployment

The project includes a `deploy.sh` script for easy deployment to **Google Cloud Run**:

```bash
PROJECT_ID=my-project GCS_BUCKET=my-bucket ./deploy.sh
```

## 🛠️ Configuration Tuning

The application supports several tuning parameters in `app/speech.py`:
- `min_speaker_count` / `max_speaker_count`: Helps the diarization engine be more accurate.
- `denoise_audio`: Toggleable noise reduction for low-quality recordings.
- `enable_automatic_punctuation`: Automatically adds punctuation to transcripts for better readability.
