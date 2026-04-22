from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import uvicorn
import logging

from auth.routers import user_auth
from chatbot.main import (
    validation_error_handler,
    generic_error_handler,
    root,
    health,
    chat,
    voice_to_hadith,
    image_to_hadith,
    get_session_stats,
    delete_session,
    get_rate_limit_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("RAG APP")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("App (Auth + Chatbot) starting up...")
    yield
    log.info(" App shutting down...")
    try:
        from chatbot.utils import _http_session
        if _http_session and not _http_session.closed:
            await _http_session.close()
            log.info("✅ aiohttp session closed.")
    except Exception:
        pass


app = FastAPI(
    title="Hadith Chatbot + Auth API",
    version="3.0.0",
    description=(
        "Combined Authentication + Hadith Chatbot with:\n"
        "- 400K token context window management\n"
        "- Azure rate limit guard (250K tpm)\n"
        "- Conversation memory & history\n"
        "- Old vs New data tracking\n\n"
        "**Protected endpoints ke liye:** Upar Authorize 🔒 button dabao, token paste karo."
    ),
    swagger_ui_parameters={"persistAuthorization": True},
    lifespan=lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schema.setdefault("components", {})
    schema["components"]["securitySchemes"] = {
        "x-access-token": {
            "type": "apiKey",
            "in": "header",
            "name": "x-access-token",
            "description": "Login se mila hua token yahan paste karo",
        }
    }

    PROTECTED_PATHS = {
        "/chat",
        "/voice-to-hadith",
        "/image-to-hadith",
        "/auth/update_profile",
        "/auth/get_profile",
        "/auth/logout",
        "/auth/request_delete_account",
        "/auth/confirm_delete_account",
    }

    for path, path_item in schema.get("paths", {}).items():
        if path in PROTECTED_PATHS:
            for method in path_item.values():
                # Swagger mein lock icon aayega aur token header mein jayega automatically
                method["security"] = [{"x-access-token": []}]

    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

# Auth Routes
app.include_router(user_auth, prefix="/auth", tags=["Authentication"])

# Chatbot Routes
app.add_api_route("/",                root,                  methods=["GET"],    tags=["Health"])
app.add_api_route("/health",          health,                methods=["GET"],    tags=["Health"])
app.add_api_route("/chat",            chat,                  methods=["POST"],   tags=["Chatbot"])
app.add_api_route("/voice-to-hadith", voice_to_hadith,       methods=["POST"],   tags=["Chatbot"])
app.add_api_route("/image-to-hadith", image_to_hadith,       methods=["POST"],   tags=["Chatbot"])

# # Session Routes
# app.add_api_route("/session/{session_id}", get_session_stats, methods=["GET"],    tags=["Session"])
# app.add_api_route("/session/{session_id}", delete_session,    methods=["DELETE"], tags=["Session"])

# Rate Limit Monitoring
# app.add_api_route("/rate-limit",      get_rate_limit_status, methods=["GET"],    tags=["Monitoring"])


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info",
    )