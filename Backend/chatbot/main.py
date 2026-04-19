from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from fastapi.middleware.cors import CORSMiddleware

from .audio import voice_to_hadith_query
from .image import extract_hadith_from_image
from .utils import generate_response
from .context_manager import session_store, rate_tracker
from auth.jwt_dacorator import token_required

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("hadith_api")

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_AUDIO_DIR = os.getenv("UPLOAD_AUDIO_DIR", "uploads")
UPLOAD_IMAGE_DIR = os.getenv("UPLOAD_IMAGE_DIR", "upload_image")
MAX_AUDIO_MB     = int(os.getenv("MAX_AUDIO_MB", "25"))
MAX_IMAGE_MB     = int(os.getenv("MAX_IMAGE_MB", "10"))
MAX_AUDIO_BYTES  = MAX_AUDIO_MB * 1024 * 1024
MAX_IMAGE_BYTES  = MAX_IMAGE_MB * 1024 * 1024

ALLOWED_AUDIO_EXT = {".mp3", ".mp4", ".wav", ".webm", ".m4a", ".ogg", ".flac"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".gif", ".webp"}

os.makedirs(UPLOAD_AUDIO_DIR, exist_ok=True)
os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Hadith API starting up...")
    yield
    log.info("Hadith API shutting down...")
    try:
        from .utils import _http_session
        if _http_session and not _http_session.closed:
            await _http_session.close()
            log.info("✅ aiohttp session closed.")
    except Exception:
        pass


# ── Global exception handlers ─────────────────────────────────────────────────
async def validation_error_handler(request: Request, exc: RequestValidationError):
    log.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": "Invalid request data.", "details": exc.errors()},
    )


async def generic_error_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error. Please try again later."},
    )


# ── Models ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query       : str
    book_filter : Optional[str] = None
    session_id  : Optional[str] = None   # NEW: optional session for context history

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Query too long (max 2000 characters).")
        return v

    @field_validator("book_filter", mode="before")
    @classmethod
    def strip_book_filter(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


async def _read_upload(file: UploadFile, max_bytes: int, allowed_ext: set[str]) -> bytes:
    ext = _ext(file.filename)
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed_ext))}",
        )
    try:
        content = await file.read()
    except Exception as exc:
        log.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(status_code=400, detail="Could not read uploaded file.")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed: {max_bytes // (1024*1024)} MB.",
        )
    return content


def _write_temp(content: bytes, filename: str, upload_dir: str) -> str:
    safe_name = f"{uuid.uuid4()}_{os.path.basename(filename or 'upload')}"
    path      = os.path.join(upload_dir, safe_name)
    try:
        with open(path, "wb") as f:
            f.write(content)
    except OSError as exc:
        log.error("Failed to write temp file %s: %s", path, exc)
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")
    return path


def _cleanup(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as exc:
        log.warning("Could not delete temp file %s: %s", path, exc)


# ── Routes ────────────────────────────────────────────────────────────────────
async def root():
    return {"status": "ok", "message": "Hadith Chat API is running!"}


async def health():
    return {
        "status": "healthy",
        "rate_limit": rate_tracker.status(),
        "active_sessions": session_store.active_sessions(),
    }


async def chat(request: ChatRequest, _token: dict = Depends(token_required)):
    """
    /chat endpoint — supports optional session_id for multi-turn conversations.

    How context management works:
    ──────────────────────────────
    1. If session_id provided → history is preserved and sent with each request
    2. Each call adds the user query + assistant reply to session memory
    3. When history gets too large → older turns are summarized automatically
    4. Key facts (names, preferences) are extracted and never lost
    5. RAG context (hadiths) is always labeled as NEW DATA

    How rate limiting works:
    ─────────────────────────
    - Before each Azure API call, we check tokens used in last 60 seconds
    - If projected usage > 220K tpm → we sleep until capacity frees up
    - Actual usage is recorded after each call
    - You can see current rate status in /health endpoint
    """
    query       = request.query
    book_filter = request.book_filter
    session_id  = request.session_id
    start       = time.perf_counter()

    log.info("📥 /chat | book_filter=%s | session=%s | query=%.80s",
             book_filter or "all", session_id or "none", query)

    try:
        result = await asyncio.wait_for(
            generate_response(query, book_filter=book_filter, session_id=session_id),
            timeout=120,
        )
    except asyncio.TimeoutError:
        log.error("generate_response timed out for query: %.80s", query)
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    except Exception as exc:
        log.exception("generate_response failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate response.")

    if not result:
        raise HTTPException(status_code=500, detail="Empty response from model.")

    response_data = {
        "success"           : True,
        "query"             : query,
        "book_filter"       : book_filter,
        "response"          : result,
        "time_taken_seconds": round(time.perf_counter() - start, 2),
        "session_id"        : session_id,
    }

    # Include session stats if session was used
    if session_id:
        session = session_store.get(session_id)
        if session:
            response_data["session_stats"] = session.stats()

    return response_data


async def voice_to_hadith(
    file        : UploadFile    = File(...),
    user_text   : Optional[str] = Form(None),
    book_filter : Optional[str] = Form(None),
    session_id  : Optional[str] = Form(None),
    _token      : dict          = Depends(token_required),
):
    start       = time.perf_counter()
    file_path   = None
    book_filter = book_filter.strip() if book_filter and book_filter.strip() else None

    log.info("📥 /voice-to-hadith | book_filter=%s | session=%s",
             book_filter or "all", session_id or "none")

    content   = await _read_upload(file, MAX_AUDIO_BYTES, ALLOWED_AUDIO_EXT)
    file_path = _write_temp(content, file.filename, UPLOAD_AUDIO_DIR)

    try:
        loop = asyncio.get_event_loop()

        try:
            voice_result = await asyncio.wait_for(
                loop.run_in_executor(None, voice_to_hadith_query, file_path),
                timeout=60,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Audio transcription timed out.")
        except Exception as exc:
            log.error("Transcription error: %s", exc)
            raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}")

        voice_text = (voice_result.get("transcribed_text") or "").strip()
        if not voice_text:
            raise HTTPException(status_code=422, detail="Could not transcribe audio. Please speak clearly.")

        ut             = (user_text or "").strip()
        combined_query = (
            f"User Text: {ut}\nVoice Text: {voice_text}" if ut else voice_text
        )

        try:
            result = await asyncio.wait_for(
                generate_response(combined_query, book_filter=book_filter, session_id=session_id),
                timeout=120,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Response generation timed out.")
        except Exception as exc:
            log.exception("generate_response failed in voice route: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to generate response.")

        return {
            "success"           : True,
            "book_filter"       : book_filter,
            "response"          : result,
            "time_taken_seconds": round(time.perf_counter() - start, 2),
            "session_id"        : session_id,
        }

    finally:
        _cleanup(file_path)


async def image_to_hadith(
    file        : UploadFile    = File(...),
    user_text   : Optional[str] = Form(None),
    book_filter : Optional[str] = Form(None),
    session_id  : Optional[str] = Form(None),
    _token      : dict          = Depends(token_required),
):
    start       = time.perf_counter()
    file_path   = None
    book_filter = book_filter.strip() if book_filter and book_filter.strip() else None

    log.info("📥 /image-to-hadith | book_filter=%s | session=%s",
             book_filter or "all", session_id or "none")

    content   = await _read_upload(file, MAX_IMAGE_BYTES, ALLOWED_IMAGE_EXT)
    file_path = _write_temp(content, file.filename, UPLOAD_IMAGE_DIR)

    try:
        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, extract_hadith_from_image, file_path),
                timeout=60,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Image processing timed out.")
        except Exception as exc:
            log.error("Image extraction error: %s", exc)
            raise HTTPException(status_code=502, detail=f"Image processing failed: {exc}")

        if not result.get("is_hadith_related"):
            return {
                "success": False,
                "message": "Image mein koi Hadith content detect nahi hua.",
            }

        optimized_query = (result.get("optimized_query") or "").strip()
        if not optimized_query:
            raise HTTPException(status_code=422, detail="Could not extract a valid query from image.")

        ut             = (user_text or "").strip()
        combined_query = f"{ut}\n{optimized_query}" if ut else optimized_query

        try:
            response = await asyncio.wait_for(
                generate_response(combined_query, book_filter=book_filter, session_id=session_id),
                timeout=120,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Response generation timed out.")
        except Exception as exc:
            log.exception("generate_response failed in image route: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to generate response.")

        return {
            "success"           : True,
            "book_filter"       : book_filter,
            "response"          : response,
            "time_taken_seconds": round(time.perf_counter() - start, 2),
            "session_id"        : session_id,
        }

    finally:
        _cleanup(file_path)


# ── Session Management Routes ─────────────────────────────────────────────────
async def get_session_stats(session_id: str):
    """Get stats for a specific session."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session.stats()


async def delete_session(session_id: str):
    """Delete / clear a session (start fresh)."""
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": f"Session {session_id} deleted successfully."}


async def get_rate_limit_status():
    """See current Azure rate limit usage."""
    return {
        "rate_limit_info": rate_tracker.status(),
        "explanation": {
            "tokens_used_last_60s": "Tokens consumed in the last 60 seconds",
            "rate_limit_tpm":       "Azure hard limit: 250,000 tokens/minute",
            "safe_limit_tpm":       "Our soft limit (12% buffer): 220,000 tokens/minute",
            "remaining_capacity":   "How many more tokens we can use right now",
            "total_tokens_served":  "All-time total tokens used since server start",
        }
    }
