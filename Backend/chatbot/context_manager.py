"""
context_manager.py
==================
Handles:
  - 400K token context window management
  - Conversation history reconstruction
  - Memory / key-facts preservation
  - Old vs New data tracking & merging
  - Azure Rate Limit: 250,000 tokens/minute awareness
  - Intelligent summarization when context overflows
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("hadith_api.context_manager")

# ══════════════════════════════════════════════════════════════════════════════
# TOKEN BUDGET CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MAX_CONTEXT_TOKENS         = 400_000   # Azure model context window
MAX_OUTPUT_TOKENS          = 128_000   # Max tokens for model output
AZURE_RATE_LIMIT_TPM       = 250_000   # Azure rate limit: tokens per minute
SAFE_RATE_LIMIT_TPM        = 220_000   # Keep 12% buffer below hard limit
HISTORY_TOKEN_BUDGET       = 80_000    # Max tokens reserved for chat history
RAG_CONTEXT_TOKEN_BUDGET   = 40_000    # Max tokens reserved for RAG context
SYSTEM_PROMPT_TOKEN_BUDGET = 4_000     # Max tokens for system prompts
USER_QUERY_TOKEN_BUDGET    = 2_000     # Max tokens for user query
MEMORY_TOKEN_BUDGET        = 8_000     # Max tokens for preserved key facts
# Remaining budget is left as safety margin

# Rough token estimator: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4

# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMIT TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class RateLimitTracker:
    """
    Tracks token usage per minute and enforces Azure's 250K TPM limit.

    HOW WE HANDLE RATE LIMITING:
    ─────────────────────────────
    1. Sliding 60-second window: we track each request's (timestamp, tokens_used).
    2. Before every API call, we sum tokens used in the last 60 seconds.
    3. If projected usage > SAFE_RATE_LIMIT_TPM (220K), we calculate how many
       seconds until enough old tokens "expire" from the window, then sleep.
    4. We log a clear message every time we wait, so you can see exactly why.
    5. Exponential back-off is applied on top if Azure still returns 429.

    This means: NO surprise RateLimitErrors from Azure under normal usage.
    """

    def __init__(self):
        # Each entry: (timestamp_float, tokens_used_int)
        self._window: deque[Tuple[float, int]] = deque()
        self._total_tokens_served = 0

    def _prune(self):
        """Remove entries older than 60 seconds."""
        now = time.time()
        while self._window and (now - self._window[0][0]) > 60:
            self._window.popleft()

    def tokens_used_last_minute(self) -> int:
        self._prune()
        return sum(t for _, t in self._window)

    def wait_if_needed(self, estimated_tokens: int) -> float:
        """
        Block (sleep) if adding estimated_tokens would exceed SAFE_RATE_LIMIT_TPM.
        Returns how many seconds we waited (0 if no wait needed).
        """
        self._prune()
        current_usage = self.tokens_used_last_minute()
        projected     = current_usage + estimated_tokens

        if projected <= SAFE_RATE_LIMIT_TPM:
            return 0.0

        # Calculate how long until enough tokens expire from the window
        overage = projected - SAFE_RATE_LIMIT_TPM
        wait_seconds = 0.0
        running = 0
        now = time.time()

        for ts, tok in self._window:
            running += tok
            if running >= overage:
                age_of_entry = now - ts
                wait_seconds = max(0.0, 60.0 - age_of_entry + 0.5)  # +0.5s buffer
                break

        if wait_seconds <= 0:
            wait_seconds = 5.0  # fallback

        log.warning(
            "⏳ RATE LIMIT GUARD | current=%d tpm | estimated=%d | projected=%d | "
            "limit=%d | waiting=%.1fs",
            current_usage, estimated_tokens, projected, SAFE_RATE_LIMIT_TPM, wait_seconds
        )
        time.sleep(wait_seconds)
        return wait_seconds

    def record(self, tokens_used: int):
        """Call this AFTER a successful API call to record token usage."""
        self._prune()
        self._window.append((time.time(), tokens_used))
        self._total_tokens_served += tokens_used
        log.debug(
            "📊 Rate tracker | recorded=%d | last_min_total=%d | all_time=%d",
            tokens_used, self.tokens_used_last_minute(), self._total_tokens_served
        )

    def status(self) -> Dict[str, Any]:
        return {
            "tokens_used_last_60s": self.tokens_used_last_minute(),
            "rate_limit_tpm":       AZURE_RATE_LIMIT_TPM,
            "safe_limit_tpm":       SAFE_RATE_LIMIT_TPM,
            "total_tokens_served":  self._total_tokens_served,
            "remaining_capacity":   SAFE_RATE_LIMIT_TPM - self.tokens_used_last_minute(),
        }


# Singleton — shared across all requests
rate_tracker = RateLimitTracker()


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    role:      str          # "user" | "assistant"
    content:   str
    timestamp: float = field(default_factory=time.time)
    tokens:    int   = 0    # estimated token count

    def estimate_tokens(self) -> int:
        self.tokens = max(1, len(self.content) // CHARS_PER_TOKEN)
        return self.tokens


@dataclass
class MemoryFact:
    """
    A preserved key fact that must survive context summarization.
    Examples: user name, language preference, confirmed hadith references.
    """
    key:       str
    value:     str
    source:    str = "auto"    # "auto" | "user" | "system"
    timestamp: float = field(default_factory=time.time)

    def to_text(self) -> str:
        return f"{self.key}: {self.value}"


@dataclass
class ContextSnapshot:
    """
    Represents the full assembled context for ONE API call.
    Tracks what is OLD vs NEW for transparency.
    """
    system_prompt:    str
    old_history:      List[Dict]   # summarized / older turns
    new_history:      List[Dict]   # recent turns kept verbatim
    memory_block:     str          # preserved key facts
    rag_context:      str          # retrieved hadith context (NEW data)
    user_query:       str
    total_tokens:     int
    old_tokens:       int
    new_tokens:       int
    was_summarized:   bool = False


# ══════════════════════════════════════════════════════════════════════════════
# SESSION CONTEXT MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class SessionContextManager:
    """
    Per-conversation context manager.

    Responsibilities:
      1. Store all user + assistant messages
      2. Extract and preserve key memory facts
      3. Assemble full context for each API call within token budget
      4. Summarize old history when needed
      5. Clearly tag OLD vs NEW data in the assembled context
    """

    def __init__(self, session_id: str):
        self.session_id   = session_id
        self.history:     List[Message]    = []
        self.memory_facts: Dict[str, MemoryFact] = {}
        self._created_at  = time.time()
        self._call_count  = 0
        log.info("🆕 Session created: %s", session_id)

    # ── Storage ───────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str):
        """Store a message (user input OR assistant output)."""
        msg = Message(role=role, content=content)
        msg.estimate_tokens()
        self.history.append(msg)
        self._auto_extract_memory(role, content)
        log.debug("💬 [%s] +%s msg (%d tok) | history_len=%d",
                  self.session_id, role, msg.tokens, len(self.history))

    def remember(self, key: str, value: str, source: str = "user"):
        """Manually store an important fact that must survive summarization."""
        self.memory_facts[key.lower()] = MemoryFact(key=key, value=value, source=source)
        log.info("🧠 Memory stored [%s]: %s = %s", self.session_id, key, value)

    # ── Auto Memory Extraction ────────────────────────────────────────────────

    def _auto_extract_memory(self, role: str, content: str):
        """
        Heuristically extract key facts from messages.
        Keeps: names, language mentions, book preferences, verified hadiths.
        """
        if role != "user":
            return
        text = content.lower()

        # Name detection
        for phrase in ("my name is ", "i am ", "call me "):
            if phrase in text:
                idx = text.index(phrase) + len(phrase)
                name_candidate = content[idx:idx+30].split()[0].strip(".,!?")
                if name_candidate and len(name_candidate) > 1:
                    self.remember("user_name", name_candidate, "auto")

        # Language preference
        if "respond in urdu" in text or "reply in urdu" in text:
            self.remember("language", "urdu", "auto")
        elif "respond in arabic" in text:
            self.remember("language", "arabic", "auto")
        elif "respond in english" in text:
            self.remember("language", "english", "auto")

        # Book filter preference
        from chatbot.utils import BOOK_ALIASES
        for alias, canonical in BOOK_ALIASES.items():
            if alias in text and "only" in text:
                self.remember("preferred_book", canonical, "auto")
                break

    # ── Token Estimation ──────────────────────────────────────────────────────

    @staticmethod
    def _estimate(text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)

    # ── History Summarization ─────────────────────────────────────────────────

    def _summarize_old_history(self, messages: List[Message]) -> str:
        """
        Create a compact summary of older conversation turns.
        Used when full history exceeds HISTORY_TOKEN_BUDGET.
        """
        if not messages:
            return ""

        lines = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            # Keep only first 200 chars of each old message for summary
            snippet = msg.content[:200].replace("\n", " ")
            if len(msg.content) > 200:
                snippet += "..."
            lines.append(f"[{prefix}]: {snippet}")

        summary = (
            "=== SUMMARIZED OLDER CONVERSATION ===\n"
            "(These are condensed older turns — key facts are in MEMORY section)\n"
            + "\n".join(lines) +
            "\n=== END SUMMARY ==="
        )
        log.info("📦 Summarized %d old messages into %d chars [%s]",
                 len(messages), len(summary), self.session_id)
        return summary

    # ── Context Assembly ──────────────────────────────────────────────────────

    def build_context(
        self,
        user_query:  str,
        rag_context: str,
        system_prompt: str,
    ) -> ContextSnapshot:
        """
        Assemble the full prompt context for one API call.

        ASSEMBLY ORDER (priority, high → low):
          1. System prompt (always kept)
          2. Memory facts (always kept)
          3. RAG context / new data (always kept — it's the NEW data)
          4. Recent history verbatim (kept as much as budget allows)
          5. Older history summarized (remaining budget)
          6. User query (always kept)

        OLD vs NEW tracking:
          - OLD: previously stored conversation turns
          - NEW: current user query + current RAG retrieval
        """
        self._call_count += 1

        sys_tokens   = self._estimate(system_prompt)
        query_tokens = self._estimate(user_query)
        rag_tokens   = self._estimate(rag_context)

        # Memory block
        memory_lines = [f.to_text() for f in self.memory_facts.values()]
        memory_block = ""
        if memory_lines:
            memory_block = (
                "=== 🧠 PRESERVED MEMORY (Important facts — do not ignore) ===\n"
                + "\n".join(memory_lines) +
                "\n=== END MEMORY ==="
            )
        mem_tokens = self._estimate(memory_block)

        # Budget remaining for history
        overhead   = sys_tokens + query_tokens + rag_tokens + mem_tokens
        hist_budget = max(0, HISTORY_TOKEN_BUDGET - overhead)

        # Split history: recent (verbatim) vs old (summarized)
        recent_messages: List[Message] = []
        old_messages:    List[Message] = []

        accumulated = 0
        for msg in reversed(self.history):
            t = msg.tokens or msg.estimate_tokens()
            if accumulated + t <= hist_budget:
                recent_messages.insert(0, msg)
                accumulated += t
            else:
                old_messages.insert(0, msg)

        was_summarized = bool(old_messages)
        old_summary    = self._summarize_old_history(old_messages) if old_messages else ""
        old_tokens     = self._estimate(old_summary)

        # Build formatted history lists for API
        new_history_dicts = [
            {"role": m.role, "content": m.content}
            for m in recent_messages
        ]
        old_history_dicts = (
            [{"role": "system", "content": old_summary}]
            if old_summary else []
        )

        # Total token estimate
        new_tokens   = accumulated
        total_tokens = overhead + new_tokens + old_tokens

        snapshot = ContextSnapshot(
            system_prompt  = system_prompt,
            old_history    = old_history_dicts,
            new_history    = new_history_dicts,
            memory_block   = memory_block,
            rag_context    = rag_context,
            user_query     = user_query,
            total_tokens   = total_tokens,
            old_tokens     = old_tokens,
            new_tokens     = new_tokens,
            was_summarized = was_summarized,
        )

        log.info(
            "📐 Context built [%s] | call#%d | total≈%d tok | "
            "recent=%d msgs (%d tok) | old=%d msgs→summary (%d tok) | "
            "rag=%d tok | mem=%d tok | summarized=%s",
            self.session_id, self._call_count,
            total_tokens,
            len(recent_messages), new_tokens,
            len(old_messages), old_tokens,
            rag_tokens, mem_tokens,
            was_summarized,
        )
        return snapshot

    def build_messages_for_api(
        self,
        user_query:    str,
        rag_context:   str,
        system_prompt: str,
    ) -> List[Dict]:
        """
        Returns the final messages array to pass directly to Azure OpenAI.

        Structure:
          [system] → [old_summary?] → [memory?] → [rag_context] → [history] → [user_query]
        """
        snapshot = self.build_context(user_query, rag_context, system_prompt)

        messages: List[Dict] = []

        # 1. System prompt
        messages.append({"role": "system", "content": snapshot.system_prompt})

        # 2. Old history summary (OLD DATA label)
        for m in snapshot.old_history:
            messages.append(m)

        # 3. Memory block (always preserved)
        if snapshot.memory_block:
            messages.append({
                "role": "system",
                "content": snapshot.memory_block
            })

        # 4. RAG context (NEW DATA label)
        if snapshot.rag_context:
            labeled_rag = (
                "=== 📥 NEW DATA: Retrieved Hadith Context (current request) ===\n"
                + snapshot.rag_context +
                "\n=== END NEW DATA ==="
            )
            messages.append({"role": "system", "content": labeled_rag})

        # 5. Recent history verbatim (OLD conversation turns)
        if snapshot.new_history:
            messages.append({
                "role": "system",
                "content": "=== 📜 RECENT CONVERSATION HISTORY ==="
            })
            for m in snapshot.new_history:
                messages.append(m)

        # 6. Current user query (NEW DATA)
        messages.append({
            "role": "user",
            "content": (
                "=== 🆕 NEW USER QUERY ===\n"
                + user_query
            )
        })

        return messages

    def stats(self) -> Dict[str, Any]:
        total_hist_tokens = sum(m.tokens for m in self.history)
        return {
            "session_id":       self.session_id,
            "total_messages":   len(self.history),
            "memory_facts":     len(self.memory_facts),
            "history_tokens":   total_hist_tokens,
            "api_calls":        self._call_count,
            "age_seconds":      int(time.time() - self._created_at),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STORE  (in-memory, per-process)
# ══════════════════════════════════════════════════════════════════════════════

class SessionStore:
    """
    Holds all active SessionContextManager instances.
    Provides create / get / delete operations.
    """

    def __init__(self, max_sessions: int = 10_000):
        self._sessions:    Dict[str, SessionContextManager] = {}
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> SessionContextManager:
        if session_id not in self._sessions:
            if len(self._sessions) >= self._max_sessions:
                # Evict oldest session
                oldest = min(self._sessions.values(), key=lambda s: s._created_at)
                del self._sessions[oldest.session_id]
                log.warning("♻️  Session evicted (store full): %s", oldest.session_id)
            self._sessions[session_id] = SessionContextManager(session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[SessionContextManager]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            log.info("🗑️  Session deleted: %s", session_id)
            return True
        return False

    def active_sessions(self) -> int:
        return len(self._sessions)


# Singleton store — shared across all requests
session_store = SessionStore()
