import logging
import os
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from pymongo import MongoClient
from dotenv import load_dotenv
from .functions import JWT_SECRET_KEY

load_dotenv()

log = logging.getLogger("user_auth.jwt")

# DB connection — logout check ke liye
_client = MongoClient(os.getenv("MONGO_URI"))
_db = _client[os.getenv("DB_NAME", "db_name")]
_user_col = _db["user_collection_name"]

api_key_header = APIKeyHeader(name="x-access-token", auto_error=True)


async def token_required(token: str = Security(api_key_header)) -> dict:

    # ─── Step 1: JWT signature verify karo ───────────────────────────────────
    try:
        data = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={"error": "Token expired! Please login again."},
        )
    except InvalidTokenError as e:
        log.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=401,
            detail={"error": "Token is invalid. Please login again."},
        )

    # ─── Step 2: DB mein check — user ne logout to nahi kiya ─────────────────
    # Login pe token DB mein save hota hai
    # Logout pe token DB se hata diya jata hai ($unset)
    # Agar DB mein token nahi = logged out user hai
    email = data.get("email")
    if email:
        try:
            user = _user_col.find_one(
                {"email": email},
                {"token": 1}  # sirf token field — fast query
            )
            if not user or not user.get("token"):
                log.warning(f"Logged out user tried to access API: email={email}")
                raise HTTPException(
                    status_code=401,
                    detail={"error": "You are logged out. Please login again."},
                )
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"DB check error in token_required: {e}")
            pass  # DB error pe block mat karo

    log.debug("Token verified for: %s", email)
    return data