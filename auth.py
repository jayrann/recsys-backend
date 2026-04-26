"""
auth.py
=======
JWT-based authentication for the MCRS backend.

Provides:
  - hash_password(password)         — bcrypt hash
  - verify_password(plain, hashed)  — bcrypt verify
  - create_access_token(data)       — generate signed JWT
  - decode_access_token(token)      — verify and decode JWT
  - get_current_user(token)         — FastAPI dependency
  - require_admin(user)             — FastAPI dependency (admin only)
"""

from __future__ import annotations


from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from config import settings


# ─────────────────────────────────────────────────────────────────────────────
# PASSWORD HASHING (bcrypt)
# ─────────────────────────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Return bcrypt hash of the given plain-text password."""
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches hashed, False otherwise."""
    return _pwd_context.verify(plain, hashed)


# ─────────────────────────────────────────────────────────────────────────────
# JWT TOKENS
# ─────────────────────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(data: dict,
                         expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT access token.

    data should contain at minimum:
        {"sub": str(user_id), "role": "user"|"admin"}
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.token_expire_min)
    )
    to_encode["exp"] = expire
    to_encode["iat"] = datetime.utcnow()
    return jwt.encode(to_encode, settings.secret_key,
                      algorithm=settings.algorithm)


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT token.
    Returns the payload dict or None if invalid/expired.
    """
    try:
        payload = jwt.decode(token, settings.secret_key,
                             algorithms=[settings.algorithm])
        return payload
    except JWTError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

_credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

_forbidden_exception = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="Administrator access required",
)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency — validates JWT token and returns current user payload.

    Usage in route:
        @app.get("/profile/{user_id}")
        def profile(user_id: int, current=Depends(get_current_user)):
            ...
    """
    payload = decode_access_token(token)
    if payload is None:
        raise _credentials_exception

    user_id = payload.get("sub")
    role    = payload.get("role")

    if user_id is None:
        raise _credentials_exception

    return {"user_id": int(user_id), "role": role}


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """
    FastAPI dependency — requires the current user to be an administrator.

    Usage in route:
        @app.get("/admin/stats")
        def admin_stats(admin=Depends(require_admin)):
            ...
    """
    if current_user.get("role") != "admin":
        raise _forbidden_exception
    return current_user


def get_current_user_optional(
    token: str = Depends(OAuth2PasswordBearer(
        tokenUrl="/auth/login", auto_error=False
    ))
) -> Optional[dict]:
    """
    FastAPI dependency — returns current user or None if unauthenticated.
    Use for routes that work for both authenticated and anonymous users.
    """
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    return {"user_id": int(payload["sub"]), "role": payload.get("role")}
