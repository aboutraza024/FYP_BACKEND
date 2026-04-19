from __future__ import annotations
import base64
import logging
import os
from pathlib import Path
import httpx
from dotenv import load_dotenv
from .prompt import EXTRACTION_SYSTEM, QUERY_SYSTEM

load_dotenv()

log = logging.getLogger("hadith_api.image")

AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY     = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "")
CHAT_DEPLOYMENT   = os.getenv("CHAT_DEPLOYMENT", "")

_CHAT_URL: str = ""
if AZURE_ENDPOINT and CHAT_DEPLOYMENT and AZURE_API_VERSION:
    _CHAT_URL = (
        f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{CHAT_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    )
else:
    log.warning("⚠️ Chat URL could not be built — missing env vars.")

MEDIA_TYPES: dict[str, str] = {
    ".jpg" : "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png" : "image/png",
    ".gif" : "image/gif",
    ".webp": "image/webp",
}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

_HEADERS = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}

_client = httpx.Client(timeout=60)


def _encode_image(image_path: str) -> tuple[str, str]:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    size = path.stat().st_size
    if size == 0:
        raise ValueError("Image file is empty.")
    if size > MAX_IMAGE_SIZE:
        raise ValueError(f"Image too large ({size // (1024*1024)} MB). Max: 10 MB.")

    media_type = MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")

    try:
        raw     = path.read_bytes()
        encoded = base64.b64encode(raw).decode("utf-8")
        return encoded, media_type
    except OSError as exc:
        raise RuntimeError(f"Could not read image file: {exc}") from exc


def _call_api(messages: list, max_tokens: int) -> str:
    if not _CHAT_URL:
        raise RuntimeError("Image analysis service not configured. Check environment variables.")

    try:
        r = _client.post(
            _CHAT_URL,
            headers=_HEADERS,
            json={"messages": messages, "max_tokens": max_tokens, "temperature": 0},
        )
        r.raise_for_status()

        data = r.json()
        content = (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            or ""
        ).strip()

        if not content:
            raise RuntimeError("API returned an empty response.")

        return content

    except httpx.TimeoutException:
        log.error("Image API call timed out.")
        raise RuntimeError("Image analysis timed out. Please try again.")

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        body   = exc.response.text[:300]
        log.error("Image API HTTP error %d: %s", status, body)
        if status == 401:
            raise RuntimeError("Image API authentication failed. Check your API key.")
        if status == 429:
            raise RuntimeError("Image API rate limit reached. Please wait and retry.")
        if status >= 500:
            raise RuntimeError("Image analysis service is temporarily unavailable.")
        raise RuntimeError(f"Image API error {status}: {body}")

    except httpx.RequestError as exc:
        log.error("Image API network error: %s", exc)
        raise RuntimeError(f"Network error during image analysis: {exc}")

    except (KeyError, IndexError, TypeError) as exc:
        log.error("Unexpected image API response format: %s", exc)
        raise RuntimeError("Unexpected response from image analysis service.")


def extract_hadith_from_image(image_path: str) -> dict:
    """
    Extract Hadith content from image.
    Returns a structured dict — raises on unrecoverable errors.
    """
    encoded, media_type = _encode_image(image_path)
    log.info("🖼️ Analyzing image: %s", os.path.basename(image_path))

    # Step 1: Extract hadith content from image
    try:
        extracted = _call_api(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {
                    "role"   : "user",
                    "content": [
                        {
                            "type"     : "image_url",
                            "image_url": {
                                "url"   : f"data:{media_type};base64,{encoded}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "Extract only the Hadith content from this image."},
                    ],
                },
            ],
            max_tokens=1500,
        )
    except Exception as exc:
        log.error("Image extraction API call failed: %s", exc)
        raise

    if extracted.strip() == "NOT_HADITH":
        log.info("ℹ️ No Hadith content found in image.")
        return {"is_hadith_related": False, "hadith_content": None, "optimized_query": None}

    # Step 2: Optimize the extracted text into a search query
    try:
        optimized = _call_api(
            messages=[
                {"role": "system", "content": QUERY_SYSTEM},
                {"role": "user",   "content": extracted},
            ],
            max_tokens=200,
        )
    except Exception as exc:
        # Non-fatal: use raw extracted text as fallback query
        log.warning("Query optimization failed, using raw extraction: %s", exc)
        optimized = extracted[:500]

    return {
        "is_hadith_related": True,
        "hadith_content"   : extracted,
        "optimized_query"  : optimized,
    }
