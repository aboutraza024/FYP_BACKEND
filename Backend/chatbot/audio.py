
from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .prompt import AUDIO_QUERY_SYSTEM

load_dotenv()

log = logging.getLogger("hadith_api.audio")

AZURE_ENDPOINT        = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY         = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION     = os.getenv("AZURE_API_VERSION", "")
TRANSCRIBE_DEPLOYMENT = os.getenv("TRANSCRIBE_DEPLOYMENT", "")

_TRANSCRIBE_URL: str = ""
if AZURE_ENDPOINT and TRANSCRIBE_DEPLOYMENT and AZURE_API_VERSION:
    _TRANSCRIBE_URL = (
        f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{TRANSCRIBE_DEPLOYMENT}/audio/transcriptions"
        f"?api-version={AZURE_API_VERSION}"
    )
else:
    log.warning("⚠️ Transcription URL could not be built — missing env vars.")

AUDIO_TYPES: dict[str, str] = {
    ".mp3" : "audio/mpeg",
    ".mp4" : "audio/mp4",
    ".wav" : "audio/wav",
    ".webm": "audio/webm",
    ".m4a" : "audio/mp4",
    ".ogg" : "audio/ogg",
    ".flac": "audio/flac",
}
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB — Azure Whisper hard limit

_client = httpx.Client(timeout=60)


def _transcribe_audio(audio_path: str) -> str:
    if not _TRANSCRIBE_URL:
        raise RuntimeError("Transcription service is not configured. Check environment variables.")

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError("Audio file is empty.")
    if file_size > MAX_AUDIO_SIZE:
        raise ValueError(f"Audio file too large ({file_size // (1024*1024)} MB). Max: 25 MB.")

    media_type = AUDIO_TYPES.get(path.suffix.lower(), "audio/mpeg")

    try:
        with open(path, "rb") as f:
            response = _client.post(
                _TRANSCRIBE_URL,
                headers={"api-key": AZURE_API_KEY},
                files={"file": (path.name, f, media_type)},
                data={"response_format": "text"},
            )
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            raise RuntimeError("Transcription returned empty text. Please try again.")
        return text

    except httpx.TimeoutException:
        log.error("Transcription timeout for file: %s", audio_path)
        raise RuntimeError("Transcription timed out. File may be too large or the service is slow.")

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        body   = exc.response.text[:300]
        log.error("Transcription HTTP error %d: %s", status, body)
        if status == 401:
            raise RuntimeError("Transcription authentication failed. Check your API key.")
        if status == 429:
            raise RuntimeError("Transcription rate limit reached. Please wait and try again.")
        if status >= 500:
            raise RuntimeError("Transcription service is temporarily unavailable.")
        raise RuntimeError(f"Transcription error {status}: {body}")

    except httpx.RequestError as exc:
        log.error("Transcription network error: %s", exc)
        raise RuntimeError(f"Network error during transcription: {exc}")

    except OSError as exc:
        log.error("Could not read audio file %s: %s", audio_path, exc)
        raise RuntimeError(f"Could not read audio file: {exc}")


def voice_to_hadith_query(audio_path: str) -> dict:

    try:
        transcribed = _transcribe_audio(audio_path)
        log.info("✅ Transcription successful (%d chars)", len(transcribed))
        return {
            "transcribed_text": transcribed,
            "is_islamic"      : True,
            "optimized_query" : transcribed,
            "error"           : None,
        }
    except Exception as exc:
        log.error("voice_to_hadith_query failed: %s", exc)
        raise