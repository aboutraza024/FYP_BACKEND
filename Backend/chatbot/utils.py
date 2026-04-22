"""
utils.py
========
Core Hadith search + Azure OpenAI integration.

Changes from original:
  - ask_azure() now accepts pre-built messages list (from context manager)
  - All Azure API calls go through rate_tracker.wait_if_needed() BEFORE calling
  - rate_tracker.record() is called AFTER each successful call
  - generate_response() now accepts session_id and uses SessionContextManager
  - Old logic is 100% preserved — only context window + rate limit handling added
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse

from .prompt import system_prompt1, system_prompt2
from .context_manager import (
    session_store,
    rate_tracker,
    AZURE_RATE_LIMIT_TPM,
    SAFE_RATE_LIMIT_TPM,
)

load_dotenv()

log = logging.getLogger("hadith_api.utils")

AZURE_ENDPOINT       = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY        = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION    = os.getenv("AZURE_API_VERSION", "")
CHAT_DEPLOYMENT      = os.getenv("CHAT_DEPLOYMENT", "")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "")
QDRANT_URL           = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME      = os.getenv("COLLECTION_NAME", "")

_MISSING = [k for k, v in {
    "AZURE_ENDPOINT": AZURE_ENDPOINT, "AZURE_API_KEY": AZURE_API_KEY,
    "CHAT_DEPLOYMENT": CHAT_DEPLOYMENT, "EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
    "QDRANT_URL": QDRANT_URL, "COLLECTION_NAME": COLLECTION_NAME,
}.items() if not v]
if _MISSING:
    log.warning("⚠️ Missing environment variables: %s", ", ".join(_MISSING))

_EXACT_PATTERN = re.compile(
    r"(?P<book>[a-z\s\-\'`]+?)\s+(?:hadith|hadees|hadis|number|no\.?|#)?\s*(?P<num>\d+)\s*$"
)
_WS_PATTERN = re.compile(r"\s+")
_REF_RE     = re.compile(r"hadith_reference|reference")

BOOK_ALIASES: dict[str, str] = {
    "bukhari"        : "Sahih al-Bukhari",
    "Bukhari"        : "Sahih al-Bukhari",
    "sahih bukhari"  : "Sahih al-Bukhari",
    "al-bukhari"     : "Sahih al-Bukhari",
    "muslim"         : "Sahih Muslim",
    "sahih muslim"   : "Sahih Muslim",
    "abu dawud"      : "Sunan Abi Dawud",
    "dawud"          : "Sunan Abi Dawud",
    "sunan abi dawud": "Sunan Abi Dawud",
    "tirmidhi"       : "Jami` at-Tirmidhi",
    "al-tirmidhi"    : "Jami` at-Tirmidhi",
    "jami tirmidhi"  : "Jami` at-Tirmidhi",
    "ibn majah"      : "Sunan Ibn Majah",
    "majah"          : "Sunan Ibn Majah",
    "nasai"          : "Sunan an-Nasa'i",
    "nasaai"         : "Sunan an-Nasa'i",
    "al-nasai"       : "Sunan an-Nasa'i",
    "nasa'i"         : "Sunan an-Nasa'i",
}
_SORTED_ALIASES = sorted(BOOK_ALIASES, key=len, reverse=True)


def normalize_book_filter(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.strip().lower()
    for alias in _SORTED_ALIASES:
        if alias in cleaned:
            normalized = BOOK_ALIASES[alias]
            if normalized != raw.strip():
                log.info("📚 book_filter normalized: '%s' → '%s'", raw.strip(), normalized)
            return normalized
    return raw.strip()


try:
    az_client = AzureOpenAI(
        api_key        = AZURE_API_KEY,
        api_version    = AZURE_API_VERSION,
        azure_endpoint = AZURE_ENDPOINT,
    )
    log.info("✅ Azure OpenAI client initialized.")
except Exception as exc:
    log.critical("❌ Could not initialize Azure OpenAI client: %s", exc)
    az_client = None

try:
    qdrant = QdrantClient(
        url                 = QDRANT_URL,
        api_key             = QDRANT_API_KEY,
        timeout             = 180,
        check_compatibility = False,
    )
    log.info("✅ Qdrant client initialized.")
except Exception as exc:
    log.critical("❌ Could not initialize Qdrant client: %s", exc)
    qdrant = None

_http_session: Optional[aiohttp.ClientSession] = None
_SCRAPE_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_SCRAPE_TIMEOUT = aiohttp.ClientTimeout(total=20)


class CombinedPoint:
    __slots__ = ("payload",)

    def __init__(self, payload: dict):
        self.payload = payload


@dataclass
class HadithResult:
    url               : str
    success           : bool
    error             : str = field(default=None)
    scraped_reference : str = field(default="")


def _require_az_client():
    if az_client is None:
        raise RuntimeError("Azure OpenAI client is not available. Check environment variables.")


def _require_qdrant():
    if qdrant is None:
        raise RuntimeError("Qdrant client is not available. Check environment variables.")


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS  (with rate limit guard)
# ══════════════════════════════════════════════════════════════════════════════

def get_embeddings(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    _require_az_client()
    if not texts:
        return []

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Estimate tokens for this batch
        estimated_tokens = sum(len(t) // 4 for t in batch)

        wait = 2
        for attempt in range(5):
            try:
                # ── RATE LIMIT CHECK before API call ──────────────────────────
                waited = rate_tracker.wait_if_needed(estimated_tokens)
                if waited > 0:
                    log.info("⏳ Embedding: waited %.1fs for rate limit", waited)

                response   = az_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=batch)
                batch_embs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                all_embeddings.extend(batch_embs)

                # ── RECORD usage after success ─────────────────────────────────
                actual_tokens = getattr(response.usage, "total_tokens", estimated_tokens)
                rate_tracker.record(actual_tokens)
                break

            except RateLimitError:
                if attempt == 4:
                    log.error("Embedding rate limit exceeded after 5 retries.")
                    raise
                log.warning("Rate limit hit (Azure 429) — waiting %ds (attempt %d/5)...", wait, attempt + 1)
                time.sleep(wait)
                wait *= 2

            except (APITimeoutError, APIConnectionError) as exc:
                if attempt == 4:
                    log.error("Embedding connection/timeout error: %s", exc)
                    raise
                log.warning("Embedding connection error, retrying (%d/5): %s", attempt + 1, exc)
                time.sleep(wait)
                wait *= 2

            except APIStatusError as exc:
                log.error("Azure API status error during embedding: %s", exc)
                raise

            except Exception as exc:
                log.error("Unexpected embedding error: %s", exc)
                raise

    return all_embeddings


def _chunk_sort_key(point) -> int:
    cid = str(point.id)
    if "_chunk_" in cid:
        try:
            return int(cid.split("_chunk_")[-1])
        except ValueError:
            pass
    return 0


def combine_chunks(points) -> List[CombinedPoint]:
    if not points:
        return []

    grouped: dict = defaultdict(list)
    for p in points:
        try:
            bid = p.payload.get("base_id") or p.payload.get("Reference") or str(p.id)
            grouped[bid].append(p)
        except Exception as exc:
            log.warning("Skipping malformed point: %s", exc)

    combined: List[CombinedPoint] = []
    for chunks in grouped.values():
        try:
            best_chunk              = max(chunks, key=lambda p: getattr(p, "score", 0.0))
            payload                 = dict(best_chunk.payload)
            payload["page_content"] = " ".join(
                c.payload.get("page_content", "") for c in sorted(chunks, key=_chunk_sort_key)
            )
            payload["chunks_count"] = len(chunks)
            payload["search_score"] = float(getattr(best_chunk, "score", 0.0))
            combined.append(CombinedPoint(payload))
        except Exception as exc:
            log.warning("Could not combine chunk group: %s", exc)

    return combined


def detect_exact_lookup(query: str) -> tuple[Optional[str], Optional[str]]:
    try:
        q     = query.lower().strip()
        match = _EXACT_PATTERN.search(q)
        if not match:
            return None, None
        raw_book = match.group("book").strip()
        number   = match.group("num")
        for alias in _SORTED_ALIASES:
            if alias in raw_book:
                return BOOK_ALIASES[alias], number
    except Exception as exc:
        log.warning("detect_exact_lookup error: %s", exc)
    return None, None


def exact_hadith_lookup(book_name: str, hadith_number: str, final_k: int = 3) -> List[CombinedPoint]:
    _require_qdrant()
    log.info("🎯 Exact lookup: Book='%s' Number='%s'", book_name, hadith_number)

    def _scroll(must_conditions) -> list:
        try:
            rows, _ = qdrant.scroll(
                collection_name = COLLECTION_NAME,
                scroll_filter   = Filter(must=must_conditions),
                limit           = final_k * 3,
                with_payload    = True,
                with_vectors    = False,
            )
            return rows or []
        except UnexpectedResponse as exc:
            log.error("Qdrant scroll UnexpectedResponse: %s", exc)
            return []
        except Exception as exc:
            log.error("Qdrant scroll error: %s", exc)
            return []

    results = _scroll([
        FieldCondition(key="Book",          match=MatchValue(value=book_name)),
        FieldCondition(key="hadith_number", match=MatchValue(value=hadith_number)),
    ])

    if not results:
        log.info("⚠️ Exact match not found — fallback to Reference field...")
        results = _scroll([
            FieldCondition(key="Reference", match=MatchValue(value=f"{book_name} {hadith_number}")),
        ])

    if not results:
        log.info("❌ No exact match found.")
        return []

    log.info("✅ Exact lookup found %d chunk(s)", len(results))
    combined: List[CombinedPoint] = []
    for r in results[:final_k]:
        try:
            combined.append(CombinedPoint({**r.payload, "search_score": 1.0, "lookup_type": "exact"}))
        except Exception as exc:
            log.warning("Skipping malformed exact-lookup result: %s", exc)
    return combined


def search_hadiths(
    query          : str,
    top_k          : int           = 20,
    final_k        : int           = 3,
    book_filter    : Optional[str] = None,
    score_threshold: float         = 0.20,
) -> List[CombinedPoint]:
    _require_qdrant()

    exact_book, exact_num = detect_exact_lookup(query)
    if exact_book and exact_num:
        results = exact_hadith_lookup(exact_book, exact_num, final_k)
        if results:
            return results
        log.info("⚠️ Exact lookup failed — falling back to semantic search...")

    try:
        query_vector = get_embeddings([query])[0]
    except Exception as exc:
        log.error("Failed to get query embedding: %s", exc)
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    must = []
    resolved_book = normalize_book_filter(book_filter)
    if resolved_book:
        must.append(FieldCondition(key="Book", match=MatchValue(value=resolved_book)))
    search_filter = Filter(must=must) if must else None

    def _query(limit, threshold=None):
        kwargs = dict(
            collection_name = COLLECTION_NAME,
            query           = query_vector,
            limit           = limit,
            query_filter    = search_filter,
            with_payload    = True,
        )
        if threshold is not None:
            kwargs["score_threshold"] = threshold
        try:
            return qdrant.query_points(**kwargs).points or []
        except UnexpectedResponse as exc:
            log.error("Qdrant query UnexpectedResponse: %s", exc)
            return []
        except Exception as exc:
            log.error("Qdrant query error: %s", exc)
            return []

    results = _query(top_k, score_threshold)

    if not results:
        log.info("⚠️ No results above threshold — retrying without score filter...")
        results = _query(final_k)
        if not results:
            log.info("❌ No results from Qdrant.")
            return []

    try:
        combined = sorted(combine_chunks(results), key=lambda x: x.payload["search_score"], reverse=True)
        top      = combined[:final_k]
    except Exception as exc:
        log.error("Error combining chunks: %s", exc)
        return []

    log.info("✅ %d chunks → %d unique hadiths → top %d", len(results), len(combined), len(top))
    log.info("   Scores: %s", [round(p.payload.get("search_score", 0), 3) for p in top])
    return top


# ══════════════════════════════════════════════════════════════════════════════
# QUERY REWRITER  (with rate limit guard)
# ══════════════════════════════════════════════════════════════════════════════

def rewrite_query(user_query: str) -> str:
    log.info("✏️  Original: %s", user_query[:200])

    if all(detect_exact_lookup(user_query)):
        log.info("✏️  [Skipped — exact lookup detected]")
        return user_query

    if az_client is None:
        log.warning("Azure client unavailable — skipping query rewrite.")
        return user_query

    estimated_tokens = len(user_query) // 4 + 500  # prompt overhead estimate

    try:
        # ── RATE LIMIT CHECK ──────────────────────────────────────────────────
        waited = rate_tracker.wait_if_needed(estimated_tokens)
        if waited > 0:
            log.info("⏳ Query rewrite: waited %.1fs for rate limit", waited)

        resp = az_client.chat.completions.create(
            model       = CHAT_DEPLOYMENT,
            messages    = [
                {"role": "system", "content": system_prompt1},
                {"role": "user",   "content": f"Optimize:\n{user_query}"},
            ],
            temperature           = 0.2,
            max_completion_tokens = 2000,
            timeout               = 30,
        )
        rewritten = (resp.choices[0].message.content or "").strip()

        # ── RECORD usage ──────────────────────────────────────────────────────
        actual_tokens = getattr(resp.usage, "total_tokens", estimated_tokens)
        rate_tracker.record(actual_tokens)

        if not rewritten:
            log.warning("Query rewriter returned empty string — using original.")
            return user_query
        log.info("✏️  Optimized: %s", rewritten[:100])
        return rewritten

    except (APITimeoutError, APIConnectionError) as exc:
        log.warning("Query rewrite connection/timeout error: %s — using original.", exc)
        return user_query
    except RateLimitError as exc:
        log.warning("Query rewrite rate limit (Azure 429): %s — using original.", exc)
        return user_query
    except Exception as exc:
        log.warning("Query rewrite unexpected error: %s — using original.", exc)
        return user_query


def parse_hadith(html: str, url: str) -> HadithResult:
    try:
        soup    = BeautifulSoup(html, "lxml")
        ref_div = soup.find("div", class_=_REF_RE) or soup.find("span", class_=_REF_RE)
        scraped_ref = (
            _WS_PATTERN.sub(" ", ref_div.get_text(separator=" ", strip=True)).strip()
            if ref_div else ""
        )
        log.debug("🕸️  Parsed | url=%s | ref='%s'", url, scraped_ref)
        return HadithResult(url=url, success=True, scraped_reference=scraped_ref)

    except Exception as exc:
        log.warning("HTML parse error for %s: %s", url, exc)
        return HadithResult(url=url, success=False, error=str(exc))


def _verify_reference(rag_point: CombinedPoint, scraped: HadithResult) -> str:
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    rag_ref = str(rag_point.payload.get("Reference", "")).strip()
    sc_ref  = scraped.scraped_reference.strip()
    matched = bool(sc_ref) and _norm(rag_ref) == _norm(sc_ref)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("🔍 SUNNAH.COM VERIFICATION | %s", scraped.url)
    log.info("   RAG dataset  : %s", rag_ref or "(empty)")
    log.info("   Sunnah.com   : %s", sc_ref  or "(empty)")
    log.info("   Match        : %s", "✅ VERIFIED" if matched else "❌ NOT VERIFIED")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return (
        f"✅ VERIFIED | RAG: '{rag_ref}' | Sunnah: '{sc_ref}'"
        if matched else
        f"❌ NOT VERIFIED | RAG: '{rag_ref}' | Sunnah: '{sc_ref}'"
    )


def _build_context(search_results: list, scraped_results: list) -> str:
    scraped_map: dict[str, HadithResult] = {
        getattr(s, "url", ""): s for s in scraped_results
    }
    parts: List[str] = []

    for i, point in enumerate(search_results, start=1):
        try:
            p          = point.payload
            hadith_url = p.get("URL", "").strip()

            scraped = scraped_map.get(hadith_url)
            if scraped and scraped.success:
                badge = _verify_reference(point, scraped)
            elif scraped and not scraped.success:
                badge = f"⚠️ Scrape failed: {scraped.error}"
            else:
                badge = "⚠️ Not checked"

            parts.append(
                f"--- HADITH {i} ---\n"
                f"Book              : {p.get('Book', 'N/A')}\n"
                f"Reference         : {p.get('Reference', 'N/A')}\n"
                f"In-book Reference : {p.get('In-book_reference', 'N/A')}\n"
                f"Grade             : {p.get('Grade', 'N/A')}\n"
                f"URL               : {hadith_url}\n"
                f"Lookup Type       : {p.get('lookup_type', 'semantic')}\n"
                f"Sunnah.com        : {badge}\n"
                f"Arabic Text:\n{p.get('Arabic_Text', '')}\n"
                f"English Text:\n{p.get('full_hadith_text', p.get('page_content', 'N/A'))}"
            )
        except Exception as exc:
            log.warning("Could not build context for hadith %d: %s", i, exc)

    return "\n\n".join(parts)


_NO_HADITH_RESPONSE = (
    "⚠️ **Disclaimer:** *AI-generated. Not a formal Islamic Fatwa.*\n\n"
    "No relevant Hadiths were found for your query. "
    "Please rephrase or consult a qualified Islamic scholar."
)

_SERVICE_UNAVAILABLE = (
    "⚠️ **Service Temporarily Unavailable**\n\n"
    "The AI service is currently unavailable. Please try again in a moment."
)


# ══════════════════════════════════════════════════════════════════════════════
# ASK AZURE  (context-aware, rate-limited)
# ══════════════════════════════════════════════════════════════════════════════

def ask_azure(
    user_query:     str,
    search_results: list,
    scraped_results: list = [],
    session_id:     Optional[str] = None,
) -> str:
    """
    Core LLM call.

    If session_id is provided → uses SessionContextManager to build
    full context (history + memory + RAG + current query).

    If no session_id → original stateless behavior (backward compatible).
    """
    if not search_results:
        return _NO_HADITH_RESPONSE
    if az_client is None:
        log.error("Azure client unavailable in ask_azure.")
        return _SERVICE_UNAVAILABLE

    rag_context = _build_context(search_results, scraped_results)

    # ── Build messages with or without context manager ──────────────────────
    if session_id:
        session = session_store.get_or_create(session_id)
        # Store user message NOW (once) before building context
        session.add_message("user", user_query)
        log.info("📥 User message stored in session [%s]", session_id)
        messages = session.build_messages_for_api(
            user_query    = user_query,
            rag_context   = rag_context,
            system_prompt = system_prompt2,
        )
        log.info("📋 Using session context [%s] | %d messages", session_id, len(messages))
    else:
        # Original stateless behavior
        messages = [
            {"role": "system", "content": system_prompt2},
            {"role": "user",   "content": f"User Question:\n{user_query}\n\nHadith Context:\n{rag_context}"},
        ]

    # Token estimate for rate limiting
    estimated_tokens = sum(len(m.get("content","")) // 4 for m in messages) + 8000  # +output budget

    wait = 2
    for attempt in range(3):
        try:
            # ── RATE LIMIT CHECK ──────────────────────────────────────────────
            waited = rate_tracker.wait_if_needed(estimated_tokens)
            if waited > 0:
                log.info("⏳ ask_azure: waited %.1fs for rate limit [session=%s]",
                         waited, session_id or "stateless")

            resp = az_client.chat.completions.create(
                model                 = CHAT_DEPLOYMENT,
                messages              = messages,
                temperature           = 0.2,
                max_completion_tokens = 8000,
                timeout               = 90,
            )
            answer = (resp.choices[0].message.content or "").strip()

            # ── RECORD usage ──────────────────────────────────────────────────
            actual_tokens = getattr(resp.usage, "total_tokens", estimated_tokens)
            rate_tracker.record(actual_tokens)
            log.info(
                "✅ ask_azure OK | tokens=%d | rate_last_min=%d/%d tpm [session=%s]",
                actual_tokens,
                rate_tracker.tokens_used_last_minute(),
                AZURE_RATE_LIMIT_TPM,
                session_id or "stateless",
            )

            if not answer:
                log.warning("LLM returned empty response.")
                return _NO_HADITH_RESPONSE

            # ── Store only assistant reply in session (user already stored above) ──
            if session_id:
                session = session_store.get(session_id)
                if session:
                    session.add_message("assistant", answer)

            return answer

        except RateLimitError as exc:
            if attempt == 2:
                log.error("LLM rate limit exceeded after 3 retries (Azure 429): %s", exc)
                return _SERVICE_UNAVAILABLE
            log.warning("LLM rate limit (Azure 429) — waiting %ds...", wait)
            time.sleep(wait)
            wait *= 2

        except (APITimeoutError, APIConnectionError) as exc:
            if attempt == 2:
                log.error("LLM connection/timeout error: %s", exc)
                return _SERVICE_UNAVAILABLE
            log.warning("LLM connection error, retry %d/3: %s", attempt + 1, exc)
            time.sleep(wait)
            wait *= 2

        except APIStatusError as exc:
            log.error("Azure API status error in ask_azure: %s", exc)
            return _SERVICE_UNAVAILABLE

        except Exception as exc:
            log.exception("Unexpected error in ask_azure: %s", exc)
            return _SERVICE_UNAVAILABLE

    return _SERVICE_UNAVAILABLE


async def _get_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session


async def fetch_one(
    session  : aiohttp.ClientSession,
    url      : str,
    semaphore: asyncio.Semaphore,
) -> HadithResult:
    if not url or not url.startswith("http"):
        return HadithResult(url=url, success=False, error="Invalid URL")

    async with semaphore:
        for attempt in range(1, 4):
            try:
                async with session.get(url, headers=_SCRAPE_HEADERS, timeout=_SCRAPE_TIMEOUT) as resp:
                    if resp.status != 200:
                        log.warning("HTTP %d for %s", resp.status, url)
                        return HadithResult(url=url, success=False, error=f"HTTP {resp.status}")
                    html = await resp.text(encoding="utf-8", errors="replace")
                    return parse_hadith(html, url)

            except asyncio.TimeoutError:
                log.warning("Scrape timeout for %s (attempt %d/3)", url, attempt)
            except aiohttp.ClientConnectionError as exc:
                log.warning("Scrape connection error for %s: %s (attempt %d/3)", url, exc, attempt)
            except aiohttp.ClientError as exc:
                log.warning("Scrape client error for %s: %s", url, exc)
                return HadithResult(url=url, success=False, error=str(exc))
            except Exception as exc:
                log.warning("Unexpected scrape error for %s: %s", url, exc)
                return HadithResult(url=url, success=False, error=str(exc))

            if attempt < 3:
                await asyncio.sleep(2 * attempt)

    return HadithResult(url=url, success=False, error="Max retries reached")


async def fetch_all(links: List[str]) -> List[HadithResult]:
    if not links:
        return []
    sem     = asyncio.Semaphore(5)
    session = await _get_session()
    results = await asyncio.gather(
        *[fetch_one(session, url, sem) for url in links],
        return_exceptions=True,
    )
    final: List[HadithResult] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            log.error("fetch_all exception for link[%d]: %s", i, r)
            final.append(HadithResult(url=links[i], success=False, error=str(r)))
        else:
            final.append(r)
    return final


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE RESPONSE  (main entry point — now session-aware)
# ══════════════════════════════════════════════════════════════════════════════

async def generate_response(
    user_query  : Optional[str] = None,
    book_filter : Optional[str] = None,
    session_id  : Optional[str] = None,
) -> str:
    """
    Main pipeline:
      1. Load old session context (if session_id provided)
      2. Rewrite query
      3. RAG search (NEW data)
      4. Sunnah.com verification
      5. Ask Azure with full context (old history + new RAG + user query)
      6. Store user query + assistant reply in session

    Rate limit handling:
      - Every Azure API call goes through rate_tracker.wait_if_needed()
      - Actual tokens used are recorded via rate_tracker.record()
      - Status is logged after every call
    """
    if not user_query or not user_query.strip():
        log.warning("generate_response called with empty query.")
        return ""

    start = time.perf_counter()
    loop  = asyncio.get_event_loop()

    log.info("🔍 book_filter=%s | session=%s", book_filter or "all books", session_id or "none")

    # ── Store user query in session AFTER response (done in ask_azure) ───────
    # NOTE: user message is stored AFTER we get the answer to avoid
    # it being included twice in build_messages_for_api call.

    # ── Detect & expand follow-up queries using session history ───────────────
    FOLLOWUP_PHRASES = {
        "more about it", "tell me more", "explain further", "what else",
        "continue", "elaborate", "aur batao", "more", "aur", "more details",
        "tell me more about it", "and", "go on", "please continue",
        "more about this", "expand on this", "give me more",
    }
    query_stripped = user_query.strip().lower().rstrip("?.,!")

    is_followup = query_stripped in FOLLOWUP_PHRASES or len(query_stripped) <= 20 and any(
        phrase in query_stripped for phrase in ["more", "aur", "continue", "elaborate", "tell me"]
    )

    rag_search_query = user_query  # default

    if is_followup and session_id:
        session = session_store.get(session_id)
        if session and session.history:
            # Find the last substantive user query (not a follow-up itself)
            last_topic = None
            for msg in reversed(session.history):
                if msg.role == "user":
                    prev = msg.content.strip().lower().rstrip("?.,!")
                    if prev not in FOLLOWUP_PHRASES and len(prev) > 20:
                        last_topic = msg.content.strip()
                        break
            if last_topic:
                rag_search_query = last_topic
                log.info("🔄 Follow-up detected — expanding query from history: %s", last_topic[:80])

    # ── Query rewrite ─────────────────────────────────────────────────────────
    try:
        rewritten_query = await loop.run_in_executor(None, rewrite_query, rag_search_query)
    except Exception as exc:
        log.error("rewrite_query crashed: %s — using original.", exc)
        rewritten_query = rag_search_query

    # ── RAG search (NEW data) ─────────────────────────────────────────────────
    try:
        search_results = await loop.run_in_executor(
            None, search_hadiths, rewritten_query, 10, 3, book_filter
        )
    except Exception as exc:
        log.error("search_hadiths crashed: %s", exc)
        search_results = []

    log.info("✅ Retrieved %d hadith(s) from RAG", len(search_results))

    # ── Sunnah.com verification ───────────────────────────────────────────────
    scraped_results: List[HadithResult] = []
    scraped_links = [
        p.payload.get("URL", "").strip()
        for p in search_results
        if "sunnah.com" in p.payload.get("URL", "")
    ]

    if scraped_links:
        log.info("📥 Sunnah.com verification: %d URL(s)...", len(scraped_links))
        try:
            scraped_results = await asyncio.wait_for(fetch_all(scraped_links), timeout=30)
            ok = sum(1 for r in scraped_results if r.success)
            log.info("✅ Scrape done: %d/%d successful", ok, len(scraped_links))
        except asyncio.TimeoutError:
            log.warning("⏱️ Sunnah.com scrape timed out — skipping verification.")
        except Exception as exc:
            log.error("Sunnah.com scrape failed: %s — skipping verification.", exc)

    # ── LLM call (with context management + rate limit) ───────────────────────
    try:
        final_answer = await loop.run_in_executor(
            None, ask_azure, user_query, search_results, scraped_results, session_id
        )
    except Exception as exc:
        log.exception("ask_azure crashed: %s", exc)
        final_answer = _SERVICE_UNAVAILABLE

    log.info("⏱️ Total: %.2fs | rate_status=%s", time.perf_counter() - start, rate_tracker.status())
    return final_answer
