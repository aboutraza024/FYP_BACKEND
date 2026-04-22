import os
from dotenv import load_dotenv
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
import logging

load_dotenv()
JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY")
# print(JWT_SECRET_KEY)
# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("user_auth")

# ─── JWT Config ───────────────────────────────────────────────────────────────
TOKEN_LIFETIME_DAYS = 15          # Token 15 din valid rahega
VERIFY_CODE_MINUTES = 10          # Email verify code 10 minute mein expire

# ─── Mail Config ──────────────────────────────────────────────────────────────
MAIL_SERVER   = os.getenv("MAIL_SERVER", "smtp.gmail.com")
# Port 587 (STARTTLS) pehle try hota hai, fail ho to 465 (SSL) automatically
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_SENDER   = os.getenv("MAIL_DEFAULT_SENDER")

# Password hashing context — ek jagah banao, baar baar nahi
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Plain password ko hash karo."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Password verify karo."""
    return pwd_context.verify(plain, hashed)


def generate_token(email: str) -> str:
    """JWT token banao — 15 din valid."""
    try:
        payload = {
            "email": email,
            "exp": datetime.utcnow() + timedelta(days=TOKEN_LIFETIME_DAYS),
            "iat": datetime.utcnow(),
        }
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")
        logger.info(f"Token generated for email={email}")
        return token
    except Exception as e:
        logger.error(f"Token generation failed for email={email}: {e}")
        return None


def _build_message(to_email: str, subject: str, body: str) -> MIMEMultipart:
    """Email message object banao."""
    msg = MIMEMultipart()
    msg["From"]    = MAIL_SENDER
    msg["To"]      = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    return msg


def _try_starttls(to_email: str, msg_str: str, timeout: int) -> bool:
    """
    Method 1: SMTP + STARTTLS on port 587.
    Gmail standard approach — lekin kai hosting providers block karte hain.
    """
    logger.info(f"Trying STARTTLS (port 587) for {to_email}")
    server = smtplib.SMTP(MAIL_SERVER, 587, timeout=timeout)
    try:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.sendmail(MAIL_SENDER, to_email, msg_str)
        logger.info(f"STARTTLS (587) success for {to_email}")
        return True
    finally:
        try:
            server.quit()
        except Exception:
            pass


def _try_ssl(to_email: str, msg_str: str, timeout: int) -> bool:
    """
    Method 2: SMTP_SSL on port 465.
    Fallback — jab 587 blocked ho ya STARTTLS fail ho.
    """
    import ssl as _ssl
    logger.info(f"Trying SSL (port 465) for {to_email}")
    context = _ssl.create_default_context()
    server = smtplib.SMTP_SSL(MAIL_SERVER, 465, timeout=timeout, context=context)
    try:
        server.ehlo()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.sendmail(MAIL_SENDER, to_email, msg_str)
        logger.info(f"SSL (465) success for {to_email}")
        return True
    finally:
        try:
            server.quit()
        except Exception:
            pass


def _send_email(to_email: str, subject: str, body: str) -> bool:
    """
    Email sender — auto fallback:
      1st try: SMTP + STARTTLS  (port 587)
      2nd try: SMTP_SSL         (port 465)

    Common errors aur unke causes:
      SMTPAuthenticationError (534/535):
          → Gmail App Password required hai.
            myaccount.google.com/apppasswords pe jaao,
            App Password banao aur .env mein daalo.
            Normal Gmail password SMTP ke liye kaam NAHI karta.

      Connection timed out / unexpectedly closed:
          → Hosting provider ne outbound SMTP port block kiya hua hai.
            Port 465 automatically try karta hai yeh function.
            Agar dono fail hon to hosting provider se ports open karwao.

      SMTPSenderRefused / SMTPRecipientsRefused:
          → MAIL_SENDER ya to_email galat format mein hai.
    """

    # ─── Pre-flight checks ────────────────────────────────────────────────────
    missing = [k for k, v in {
        "MAIL_SERVER":          MAIL_SERVER,
        "MAIL_USERNAME":        MAIL_USERNAME,
        "MAIL_PASSWORD":        MAIL_PASSWORD,
        "MAIL_DEFAULT_SENDER":  MAIL_SENDER,
    }.items() if not v]

    if missing:
        logger.error(f"Email config missing in .env: {missing}")
        return False

    TIMEOUT = 60   # seconds

    msg_str = _build_message(to_email, subject, body).as_string()
    last_error = None

    # ─── Method 1: STARTTLS port 587 ─────────────────────────────────────────
    try:
        return _try_starttls(to_email, msg_str, TIMEOUT)

    except smtplib.SMTPAuthenticationError as e:
        # Auth fail — dusra port try karne se koi faida nahi, credentials galat hain
        code = e.smtp_code
        if code == 534:
            logger.error(
                f"Gmail App Password required (534). "
                f"Normal password kaam nahi karta. "
                f"myaccount.google.com/apppasswords pe App Password banao "
                f"aur MAIL_PASSWORD mein daalo."
            )
        elif code == 535:
            logger.error(
                f"Wrong credentials (535). "
                f"MAIL_USERNAME ({MAIL_USERNAME}) aur MAIL_PASSWORD .env mein check karo."
            )
        else:
            logger.error(f"SMTP auth error (code={code}): {e}")
        return False   # Auth error pe 465 try nahi — credentials theek karo pehle

    except (smtplib.SMTPException, OSError, TimeoutError) as e:
        logger.warning(f"STARTTLS (587) failed: {e} — trying SSL (465) fallback")
        last_error = e

    # ─── Method 2: SSL port 465 fallback ─────────────────────────────────────
    try:
        return _try_ssl(to_email, msg_str, TIMEOUT)

    except smtplib.SMTPAuthenticationError as e:
        code = e.smtp_code
        if code == 534:
            logger.error(
                f"Gmail App Password required (534) on port 465 too. "
                f"myaccount.google.com/apppasswords pe App Password banao."
            )
        else:
            logger.error(f"SSL (465) auth error (code={code}): {e}")
        return False

    except (smtplib.SMTPException, OSError, TimeoutError) as e:
        logger.error(
            f"Both SMTP methods failed for {to_email}. "
            f"Port 587 error: {last_error} | "
            f"Port 465 error: {e} | "
            f"Possible fix: hosting provider se outbound SMTP ports (587/465) open karwao, "
            f"ya Gmail ki jagah SendGrid/Mailgun jaise transactional email service use karo."
        )
        return False

    except Exception as e:
        logger.error(f"Unexpected error on SSL (465) for {to_email}: {e}")
        return False


def send_recovery_email(email: str, code: str) -> bool:
    subject = "Password Recovery Code"
    body = (
        f"Your password recovery code is: {code}\n\n"
        f"This code is valid for {VERIFY_CODE_MINUTES} minutes.\n"
        "If you did not request this, please ignore this email."
    )
    return _send_email(email, subject, body)


def send_verify_email_code(email: str, code: str) -> bool:
    subject = "Email Verification Code"
    body = (
        f"Your email verification code is: {code}\n\n"
        f"This code is valid for {VERIFY_CODE_MINUTES} minutes.\n"
        "If you did not request this, please ignore this email."
    )
    return _send_email(email, subject, body)


def generate_recovery_code() -> str:
    """4-digit numeric code."""
    return "".join(random.choices(string.digits, k=4))


def generate_delete_confirm_text() -> str:
    """
    Random 6-char alphanumeric string generate karo jo user ko
    rewrite karna hoga account delete confirm karne ke liye.
    """
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def send_delete_account_email(email: str, username: str, confirm_text: str) -> bool:
    subject = "Account Deletion Confirmation"
    body = (
        f"Hello {username},\n\n"
        f"We received a request to permanently delete your account.\n\n"
        f"To confirm deletion, please enter the following code exactly as shown:\n\n"
        f"    {confirm_text}\n\n"
        f"This code is valid for {VERIFY_CODE_MINUTES} minutes.\n\n"
        f"If you did NOT request this, please ignore this email — your account is safe.\n\n"
        f"Warning: Account deletion is permanent and cannot be undone."
    )
    return _send_email(email, subject, body)
