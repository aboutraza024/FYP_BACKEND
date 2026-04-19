# 🕌 Hadith Chatbot + Auth API  
**Version 3.0.0 — Combined Project**

---

## 📁 Project Structure

```
combined_project/
├── app.py                    ← 🚀 Main entry point — run this
├── requirements.txt
├── .env                      ← Your secrets (copy from .env.example)
├── .env.example              ← Template
├── uploads/                  ← Audio temp files (auto-created)
├── upload_image/             ← Image temp files (auto-created)
│
├── auth/                     ← Authentication module
│   ├── __init__.py
│   ├── functions.py          ← JWT, email, password helpers
│   ├── jwt_dacorator.py      ← JWT middleware
│   └── routers.py            ← All auth endpoints
│
└── chatbot/                  ← Hadith chatbot module
    ├── __init__.py
    ├── context_manager.py    ← ✨ NEW: Context window + rate limit system
    ├── utils.py              ← Azure OpenAI + Qdrant + RAG logic
    ├── main.py               ← Chatbot route handlers
    ├── prompt.py             ← System prompts
    ├── audio.py              ← Voice transcription
    └── image.py              ← Image extraction
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup environment
cp .env.example .env
# Edit .env with your actual credentials

# 3. Run
python app.py
# OR
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at: `http://localhost:8000`  
Swagger docs at: `http://localhost:8000/docs`

---

## 🔗 API Endpoints

### 🔐 Authentication  (`/auth/...`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Register new user |
| POST | `/auth/verify_user_email` | Verify email with code |
| POST | `/auth/resend_verify_code` | Resend email verification |
| POST | `/auth/login` | Login → returns JWT token |
| POST | `/auth/forgot_password` | Request password recovery |
| POST | `/auth/reset_password` | Reset password with code |
| POST | `/auth/update_profile` | Update name/email/password/pic |
| POST | `/auth/verify_user_email_to_update` | Verify email change |
| GET  | `/auth/get_profile?user_id=...` | Fetch user profile |

### 🤖 Chatbot

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Text query (supports session_id) |
| POST | `/voice-to-hadith` | Voice audio query |
| POST | `/image-to-hadith` | Image query |

### 📊 Monitoring & Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health + rate limit status |
| GET | `/rate-limit` | Live Azure token usage |
| GET | `/session/{id}` | View session stats |
| DELETE | `/session/{id}` | Clear session |

---

## 🧠 Context Management

### How it works

Every `/chat` request with a `session_id` uses full context management:

```
1. Load OLD context   → previous conversation turns from session
2. Add NEW input      → current query + freshly retrieved hadiths  
3. Merge intelligently → history summarized if > 80K tokens
4. Preserve memory    → names, preferences NEVER lost
5. Label OLD vs NEW   → every API message is clearly tagged
```

### Token Budget per Request

| Section | Budget |
|---------|--------|
| Context window total | 400,000 tokens |
| Chat history | 80,000 tokens |
| RAG context (hadiths) | 40,000 tokens |
| Preserved memory facts | 8,000 tokens |
| System prompt | 4,000 tokens |
| User query | 2,000 tokens |

### Using sessions

```json
POST /chat
{
  "query": "What does Islam say about prayer?",
  "session_id": "user_abc_session_1"
}
```

Same `session_id` → history is remembered across calls.  
No `session_id` → stateless (original behavior, fully backward compatible).

### Auto-extracted memory

The system automatically remembers:
- **User name** → "My name is Ahmed" → stored
- **Language preference** → "respond in Urdu" → stored  
- **Book preference** → "only from Bukhari" → stored

---

## ⚡ Rate Limiting

### Azure Limits
```
Hard limit:  250,000 tokens/minute
Our limit:   220,000 tokens/minute  (12% safety buffer)
```

### How we handle it — sliding window guard

Before **every** Azure API call:
1. Sum all tokens used in the last 60 seconds
2. If `current + estimated > 220K` → sleep until capacity frees
3. After the call → record actual tokens used

You will **never** get an unexpected `429 RateLimitError` under normal usage.

### Live monitoring

```bash
GET /rate-limit
```

Response:
```json
{
  "rate_limit_info": {
    "tokens_used_last_60s": 87340,
    "rate_limit_tpm": 250000,
    "safe_limit_tpm": 220000,
    "remaining_capacity": 132660,
    "total_tokens_served": 1245000
  }
}
```

### Log output example
```
⏳ RATE LIMIT GUARD | current=198000 tpm | estimated=25000 | projected=223000 | limit=220000 | waiting=8.3s
✅ ask_azure OK | tokens=22450 | rate_last_min=187000/250000 tpm
```

---

## 📌 Changes from Original Code

### What changed
- `chatbot/utils.py` → `ask_azure()` and `generate_response()` now accept `session_id`
- `chatbot/main.py` → All routes accept optional `session_id` form/body param
- `auth/routers.py` → Import paths fixed for package structure (`from auth.functions import ...`)
- `auth/jwt_dacorator.py` → Import path fixed

### What did NOT change
- All existing logic, prompts, RAG search, Qdrant queries — **100% unchanged**
- All auth flows — **100% unchanged**
- All Azure API call parameters — **100% unchanged**
- Session is **optional** — without it, everything works exactly as before

### What was added
- `chatbot/context_manager.py` — brand new file with:
  - `RateLimitTracker` — sliding window rate guard
  - `SessionContextManager` — per-conversation context
  - `SessionStore` — in-memory session store
- New endpoints: `/session/{id}` (GET/DELETE), `/rate-limit`
- `session_stats` field in `/chat` response (only when session_id used)

---

## 🔑 Auth Flow Summary

```
signup → verify_user_email → login → (use token in x-access-token header)
forgot_password → reset_password → login
update_profile (email change) → verify_user_email_to_update
```
