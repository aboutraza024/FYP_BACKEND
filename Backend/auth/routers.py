from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Optional
from bson import ObjectId
import os
import logging

from dotenv import load_dotenv
load_dotenv()

from .functions import (
    generate_recovery_code,
    generate_token,
    send_recovery_email,
    send_verify_email_code,
    send_delete_account_email,
    generate_delete_confirm_text,
    hash_password,
    verify_password,
    VERIFY_CODE_MINUTES,
    TOKEN_LIFETIME_DAYS,
)
from .jwt_decorator import token_required, api_key_header

# ─── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("user_auth.routers")

# ─── DB Setup ─────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["db_name"]
user_collection          = db["user_collection_name"]
verify_email_col         = db["verify_user_email"]          # signup verification
verify_email_update_col  = db["verify_user_email_update"]   # email change verification
recovery_col             = db["recovery_collection"]        # forgot password
delete_requests_col      = db["delete_requests"]            # account deletion confirmation

user_auth = APIRouter()


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class User(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str

class Login(BaseModel):
    email: EmailStr
    password: str

class Mail_verify(BaseModel):
    email: EmailStr
    code: str

class Mail_verify_to_update(BaseModel):
    email: EmailStr
    code: str
    userid: str

class ForgotPassword(BaseModel):
    email: EmailStr

class Recovery(BaseModel):
    code: str

class ResetPassword(BaseModel):
    email: EmailStr
    code: str
    new_password: str
    confirm_password: str

class ResendVerifyCode(BaseModel):
    email: EmailStr

class UpdateProfile(BaseModel):
    userid: str
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None
    confirm_password: Optional[str] = None
    profile_picture: Optional[str] = None


class LogoutRequest(BaseModel):
    userid: str


class DeleteAccountRequest(BaseModel):
    userid: str


class ConfirmDeleteAccountRequest(BaseModel):
    userid: str
    confirm_text: str


# ─── /signup ──────────────────────────────────────────────────────────────────

@user_auth.post("/signup")
async def signup(user: User):
    logger.info(f"Signup attempt: email={user.email}")

    # Password match check
    if user.password != user.confirm_password:
        logger.warning(f"Signup failed — password mismatch: email={user.email}")
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    # Duplicate email check
    if user_collection.find_one({"email": user.email}):
        logger.warning(f"Signup failed — email already exists: email={user.email}")
        raise HTTPException(status_code=409, detail="Email already in use.")

    # Send verification email PEHLE — agar mail fail ho to user save mat karo
    code = generate_recovery_code()
    mail_sent = send_verify_email_code(user.email, code)
    if not mail_sent:
        logger.error(f"Signup email send failed: email={user.email}")
        raise HTTPException(status_code=500, detail="Mail server error. Please try again.")

    # BUG FIX: plain password save nahi karo — sirf hashed
    hashed = hash_password(user.password)
    user_dict = {
        "username": user.username,
        "email": user.email,
        "password_hashed": hashed,          # sirf yahi chahiye
        "is_verified": False,
        "created_at": datetime.utcnow(),
    }
    inserted = user_collection.insert_one(user_dict)
    user_id = str(inserted.inserted_id)

    # Verification code save karo — expiry ke sath
    verify_email_col.delete_many({"email": user.email})  # purana clear karo
    verify_email_col.insert_one({
        "email": user.email,
        "code": code,
        "expires_at": datetime.utcnow() + timedelta(minutes=VERIFY_CODE_MINUTES),
    })

    logger.info(f"Signup successful — awaiting verification: email={user.email}, id={user_id}")
    return {
        "message": "User created. Please verify your email within 10 minutes.",
        "id": user_id,
        "username": user.username,
        "verify_expires_in_minutes": VERIFY_CODE_MINUTES,
    }


# ─── /resend_verify_code ──────────────────────────────────────────────────────

@user_auth.post("/resend_verify_code")
async def resend_verify_code(data: ResendVerifyCode):
    """
    BUG FIX: Galat code daalne pe user delete ho jata tha.
    Ab user resend kar sakta hai — DB mein code update ho jata hai.
    """
    logger.info(f"Resend verify code request: email={data.email}")

    user = user_collection.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email.")

    if user.get("is_verified"):
        raise HTTPException(status_code=400, detail="Email is already verified.")

    code = generate_recovery_code()
    mail_sent = send_verify_email_code(data.email, code)
    if not mail_sent:
        raise HTTPException(status_code=500, detail="Could not send email. Try again.")

    # Naya code update karo
    verify_email_col.update_one(
        {"email": data.email},
        {"$set": {
            "code": code,
            "expires_at": datetime.utcnow() + timedelta(minutes=VERIFY_CODE_MINUTES),
        }},
        upsert=True,
    )
    logger.info(f"Verification code resent: email={data.email}")
    return {
        "message": "New verification code sent.",
        "verify_expires_in_minutes": VERIFY_CODE_MINUTES,
    }


# ─── /verify_user_email ───────────────────────────────────────────────────────

@user_auth.post("/verify_user_email")
async def verify_user_email(verify: Mail_verify):
    logger.info(f"Email verify attempt: email={verify.email}")

    recovery_data = verify_email_col.find_one({"email": verify.email})

    if not recovery_data:
        logger.warning(f"Verify failed — no record found: email={verify.email}")
        raise HTTPException(status_code=400, detail="No verification pending for this email. Please signup again.")

    # BUG FIX: expiry check
    if datetime.utcnow() > recovery_data.get("expires_at", datetime.utcnow()):
        logger.warning(f"Verify failed — code expired: email={verify.email}")
        # User delete mat karo — sirf code delete karo
        verify_email_col.delete_one({"email": verify.email})
        raise HTTPException(
            status_code=400,
            detail=f"Verification code expired. Please request a new one using /resend_verify_code."
        )

    # BUG FIX: galat code pe user delete hona BAND — sirf error return karo
    if recovery_data["code"] != verify.code:
        logger.warning(f"Verify failed — wrong code: email={verify.email}")
        raise HTTPException(
            status_code=400,
            detail="Invalid code. Please try again or use /resend_verify_code."
        )

    # Sab theek — user ko verified mark karo
    user_collection.update_one(
        {"email": verify.email},
        {"$set": {"is_verified": True}}
    )
    verify_email_col.delete_one({"email": verify.email})

    user = user_collection.find_one({"email": verify.email})
    logger.info(f"Email verified successfully: email={verify.email}")
    return {
        "message": "Email verified successfully.",
        "id": str(user["_id"]),
    }


# ─── /login ───────────────────────────────────────────────────────────────────

@user_auth.post("/login")
async def login(login_data: Login):
    logger.info(f"Login attempt: email={login_data.email}")

    user = user_collection.find_one({"email": login_data.email})

    if not user or not verify_password(login_data.password, user.get("password_hashed", "")):
        logger.warning(f"Login failed — invalid credentials: email={login_data.email}")
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    # BUG FIX: Unverified user login nahi kar sakta
    if not user.get("is_verified", False):
        logger.warning(f"Login blocked — email not verified: email={login_data.email}")
        raise HTTPException(
            status_code=403,
            detail="Please verify your email first. Use /resend_verify_code if code expired."
        )

    token = generate_token(login_data.email)
    token_expiry = datetime.utcnow() + timedelta(days=TOKEN_LIFETIME_DAYS)

    user_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"token": token, "token_timestamp": token_expiry}}
    )

    logger.info(f"Login successful: email={login_data.email}, id={str(user['_id'])}")
    # NOTE: session_id backend nahi banata — yeh sirf frontend ki zimmedari hai.
    # Frontend chatbot screen par aate hi fresh uuid generate kare aur state mein rakhe.
    return {
        "userid": str(user["_id"]),
        "token": token,
    }


# ─── /forgot_password ─────────────────────────────────────────────────────────

@user_auth.post("/forgot_password")
async def forgot_password(data: ForgotPassword):
    logger.info(f"Forgot password request: email={data.email}")

    user = user_collection.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not found.")

    code = generate_recovery_code()
    mail_sent = send_recovery_email(data.email, code)
    if not mail_sent:
        logger.error(f"Recovery email failed: email={data.email}")
        raise HTTPException(status_code=500, detail="Could not send recovery email. Try again.")

    recovery_col.delete_many({"user_id": user["_id"]})   # purana code hatao
    recovery_col.insert_one({
        "user_id": user["_id"],
        "email": data.email,
        "recovery_code": code,
        "expires_at": datetime.utcnow() + timedelta(minutes=VERIFY_CODE_MINUTES),
    })

    logger.info(f"Recovery code sent: email={data.email}")
    return {
        "message": "Recovery code sent to your email.",
        "expires_in_minutes": VERIFY_CODE_MINUTES,
    }


# ─── /verify_recovery_code ────────────────────────────────────────────────────
#
# @user_auth.post("/verify_recovery_code")
# async def verify_recovery_code(recovery: Recovery):
#     logger.info(f"Recovery code verify attempt: code={recovery.code}")
#
#     recovery_data = recovery_col.find_one({"recovery_code": recovery.code})
#     if not recovery_data:
#         raise HTTPException(status_code=400, detail="Invalid recovery code.")
#
#     if datetime.utcnow() > recovery_data.get("expires_at", datetime.utcnow()):
#         recovery_col.delete_one({"_id": recovery_data["_id"]})
#         raise HTTPException(status_code=400, detail="Recovery code expired. Please request a new one.")
#
#     logger.info(f"Recovery code verified successfully")
#     return {"message": "Recovery code verified."}


# ─── /reset_password ──────────────────────────────────────────────────────────

@user_auth.post("/reset_password")
async def reset_password(data: ResetPassword):
    logger.info(f"Reset password request: email={data.email}")

    user = user_collection.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not found.")

    recovery_data = recovery_col.find_one({"recovery_code": data.code})
    if not recovery_data:
        raise HTTPException(status_code=400, detail="Invalid recovery code.")

    if datetime.utcnow() > recovery_data.get("expires_at", datetime.utcnow()):
        recovery_col.delete_one({"_id": recovery_data["_id"]})
        raise HTTPException(status_code=400, detail="Recovery code expired.")

    if data.new_password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    # BUG FIX: sirf hashed password save karo — plain password bilkul nahi
    hashed = hash_password(data.new_password)
    user_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"password_hashed": hashed},
         "$unset": {"password": "", "plain_password": ""}}   # purane plain fields hatao
    )
    recovery_col.delete_one({"_id": recovery_data["_id"]})

    logger.info(f"Password reset successful: email={data.email}")
    return {"message": "Password reset successfully. You can now login."}


# ─── /update_profile ──────────────────────────────────────────────────────────

@user_auth.post("/update_profile")
async def update_profile(profile_data: UpdateProfile, token_data: dict = Security(token_required)):
    user_id = profile_data.userid
    logger.info(f"Update profile request: userid={user_id}")

    if not user_id:
        raise HTTPException(status_code=400, detail="UserId required.")

    try:
        user = user_collection.find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID format.")

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # ─── Token ownership check ────────────────────────────────────────────────
    # Token mein jo email hai woh is userid wale user ki email se match karni chahiye
    # Warna koi dusra banda kisi aur ka profile update kar sakta tha
    token_email = token_data.get("email")
    if user.get("email") != token_email:
        logger.warning(f"Unauthorized update attempt: token_email={token_email}, userid={user_id}")
        raise HTTPException(status_code=403, detail="Unauthorized: You can only update your own profile.")

    new_data = {}

    # Profile picture
    if profile_data.profile_picture is not None:
        new_data["profile_pic"] = profile_data.profile_picture
        logger.info(f"Profile picture update: userid={user_id}")

    # Name
    if profile_data.name is not None:
        new_data["username"] = profile_data.name
        logger.info(f"Name update: userid={user_id}, name={profile_data.name}")

    # Email change — verification flow
    if profile_data.email is not None:
        code = generate_recovery_code()
        mail_sent = send_verify_email_code(profile_data.email, code)
        if not mail_sent:
            raise HTTPException(status_code=500, detail="Could not send verification email.")

        verify_email_update_col.delete_many({"userid": user_id})
        verify_email_update_col.insert_one({
            "userid": user_id,
            "new_email": profile_data.email,
            "code": code,
            "expires_at": datetime.utcnow() + timedelta(minutes=VERIFY_CODE_MINUTES),
        })
        logger.info(f"Email change code sent: userid={user_id}, new_email={profile_data.email}")

        # BUG FIX: email update early return se pehle name/pic updates bhi apply karo
        if new_data:
            user_collection.update_one({"_id": ObjectId(user_id)}, {"$set": new_data})
            logger.info(f"Other profile fields updated alongside email request: userid={user_id}")

        return JSONResponse(
            content={
                "message": "Verification email sent to new address.",
                "verify_expires_in_minutes": VERIFY_CODE_MINUTES,
            },
            status_code=200,
        )

    # Password change
    if profile_data.current_password:
        if not verify_password(profile_data.current_password, user.get("password_hashed", "")):
            logger.warning(f"Password update failed — wrong current password: userid={user_id}")
            return JSONResponse(content={"message": "Current password is incorrect."}, status_code=400)

        if profile_data.new_password != profile_data.confirm_password:
            return JSONResponse(content={"message": "New passwords do not match."}, status_code=400)

        # BUG FIX: sirf hashed save karo
        new_data["password_hashed"] = hash_password(profile_data.new_password)
        logger.info(f"Password updated: userid={user_id}")

    # BUG FIX: update apply karo — pehle sirf return ho jata tha bina update ke
    if new_data:
        result = user_collection.update_one({"_id": ObjectId(user_id)}, {"$set": new_data})
        if result.modified_count == 0:
            logger.warning(f"Update ran but nothing changed: userid={user_id}")
            return JSONResponse(content={"message": "No changes applied (data may be same)."}, status_code=200)
        logger.info(f"Profile updated successfully: userid={user_id}, fields={list(new_data.keys())}")
        return JSONResponse(content={"message": "Profile updated successfully."}, status_code=200)
    else:
        return JSONResponse(content={"message": "Nothing to update."}, status_code=200)


# ─── /verify_user_email_to_update ─────────────────────────────────────────────

@user_auth.post("/verify_user_email_to_update")
async def verify_email_to_update(verify: Mail_verify_to_update):
    logger.info(f"Email update verify: userid={verify.userid}, new_email={verify.email}")

    recovery_data = verify_email_update_col.find_one({
        "new_email": verify.email,
        "userid": verify.userid,
    })

    if not recovery_data:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code.")

    if datetime.utcnow() > recovery_data.get("expires_at", datetime.utcnow()):
        verify_email_update_col.delete_one({"_id": recovery_data["_id"]})
        raise HTTPException(status_code=400, detail="Code expired. Request email change again.")

    if recovery_data["code"] != verify.code:
        raise HTTPException(status_code=400, detail="Invalid code.")

    user_collection.update_one(
        {"_id": ObjectId(verify.userid)},
        {"$set": {"email": verify.email}}
    )
    verify_email_update_col.delete_one({"_id": recovery_data["_id"]})

    logger.info(f"Email updated successfully: userid={verify.userid}, new_email={verify.email}")
    return JSONResponse(content={"message": "Email updated successfully."}, status_code=200)


# ─── /get_profile ─────────────────────────────────────────────────────────────

@user_auth.get("/get_profile")
async def get_profile(user_id: str, token_data: dict = Security(token_required)):
    logger.info(f"Get profile request: userid={user_id}")

    if not user_id:
        raise HTTPException(status_code=400, detail="UserId required.")

    try:
        user = user_collection.find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID.")

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # ─── Token ownership check ────────────────────────────────────────────────
    token_email = token_data.get("email")
    if user.get("email") != token_email:
        logger.warning(f"Unauthorized get_profile attempt: token_email={token_email}, userid={user_id}")
        raise HTTPException(status_code=403, detail="Unauthorized: You can only view your own profile.")

    content = {
        "username": user.get("username"),
        "email": user.get("email"),
        "picture": user.get("profile_pic"),
        "is_verified": user.get("is_verified", False),
    }
    logger.info(f"Profile fetched: userid={user_id}")
    return JSONResponse(content=content, status_code=200)


# ─── /logout ──────────────────────────────────────────────────────────────────

@user_auth.post("/logout")
async def logout(data: LogoutRequest, token_data: dict = Security(token_required)):
    logger.info(f"Logout request: userid={data.userid}")

    try:
        user = user_collection.find_one({"_id": ObjectId(data.userid)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID format.")

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Token ownership check — sirf apna account logout kar sako
    token_email = token_data.get("email")
    if user.get("email") != token_email:
        logger.warning(f"Unauthorized logout attempt: token_email={token_email}, userid={data.userid}")
        raise HTTPException(status_code=403, detail="Unauthorized: You can only logout your own account.")

    # Token aur timestamp DB se hatao
    user_collection.update_one(
        {"_id": ObjectId(data.userid)},
        {"$unset": {"token": "", "token_timestamp": ""}}
    )

    logger.info(f"Logout successful: userid={data.userid}")
    return JSONResponse(content={"message": "Logged out successfully."}, status_code=200)


# ─── /request_delete_account ──────────────────────────────────────────────────
# Step 1: User delete dabata hai → email pe confirm_text bhejo

@user_auth.post("/request_delete_account")
async def request_delete_account(data: DeleteAccountRequest, token_data: dict = Security(token_required)):
    logger.info(f"Delete account request: userid={data.userid}")

    try:
        user = user_collection.find_one({"_id": ObjectId(data.userid)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID format.")

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Token ownership check
    token_email = token_data.get("email")
    if user.get("email") != token_email:
        logger.warning(f"Unauthorized delete request: token_email={token_email}, userid={data.userid}")
        raise HTTPException(status_code=403, detail="Unauthorized: You can only delete your own account.")

    # Random confirmation text generate karo
    confirm_text = generate_delete_confirm_text()

    # Email bhejo jisme confirm_text ho
    mail_sent = send_delete_account_email(
        email=user["email"],
        username=user.get("username", "User"),
        confirm_text=confirm_text,
    )
    if not mail_sent:
        logger.error(f"Delete confirm email failed: userid={data.userid}")
        raise HTTPException(status_code=500, detail="Could not send confirmation email. Try again.")

    # DB mein save karo — purana request hatao pehle
    delete_requests_col.delete_many({"userid": data.userid})
    delete_requests_col.insert_one({
        "userid": data.userid,
        "confirm_text": confirm_text,
        "expires_at": datetime.utcnow() + timedelta(minutes=VERIFY_CODE_MINUTES),
    })

    logger.info(f"Delete confirmation email sent: userid={data.userid}, email={user['email']}")
    return JSONResponse(
        content={
            "message": "Confirmation code sent to your email. Enter it exactly to confirm deletion.",
            "expires_in_minutes": VERIFY_CODE_MINUTES,
        },
        status_code=200,
    )


# ─── /confirm_delete_account ──────────────────────────────────────────────────
# Step 2: User email se mila confirm_text type karta hai → verify karo → delete

@user_auth.post("/confirm_delete_account")
async def confirm_delete_account(data: ConfirmDeleteAccountRequest, token_data: dict = Security(token_required)):
    logger.info(f"Delete account confirm attempt: userid={data.userid}")

    try:
        user = user_collection.find_one({"_id": ObjectId(data.userid)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID format.")

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Token ownership check
    token_email = token_data.get("email")
    if user.get("email") != token_email:
        logger.warning(f"Unauthorized delete confirm: token_email={token_email}, userid={data.userid}")
        raise HTTPException(status_code=403, detail="Unauthorized: You can only delete your own account.")

    # Delete request record dhundo
    delete_request = delete_requests_col.find_one({"userid": data.userid})
    if not delete_request:
        raise HTTPException(
            status_code=400,
            detail="No deletion request found. Please use /request_delete_account first.",
        )

    # Expiry check
    if datetime.utcnow() > delete_request.get("expires_at", datetime.utcnow()):
        delete_requests_col.delete_one({"_id": delete_request["_id"]})
        logger.warning(f"Delete confirm expired: userid={data.userid}")
        raise HTTPException(
            status_code=400,
            detail="Confirmation code expired. Please request deletion again.",
        )

    # Confirm text verify karo — exact match, case-sensitive
    if delete_request["confirm_text"] != data.confirm_text:
        logger.warning(f"Delete confirm wrong text: userid={data.userid}")
        raise HTTPException(
            status_code=400,
            detail="Incorrect confirmation code. Please check your email and try again.",
        )

    user_email = user["email"]

    # ─── Permanent Deletion ───────────────────────────────────────────────────
    user_collection.delete_one({"_id": ObjectId(data.userid)})
    delete_requests_col.delete_many({"userid": data.userid})

    # Related data bhi clean karo
    verify_email_col.delete_many({"email": user_email})
    verify_email_update_col.delete_many({"userid": data.userid})
    recovery_col.delete_many({"user_id": ObjectId(data.userid)})

    logger.info(f"Account permanently deleted: userid={data.userid}, email={user_email}")
    return JSONResponse(
        content={"message": "Your account has been permanently deleted."},
        status_code=200,
    )
