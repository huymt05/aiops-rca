from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .store import VALID_ROLES, count_users, create_user, create_session, get_active_session, get_user, write_audit_log


AUTH_ENABLED = os.environ.get("AIOPS_AUTH_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
SESSION_COOKIE_NAME = "aiops_session"
SESSION_TTL_HOURS = int(os.environ.get("AIOPS_SESSION_TTL_HOURS", "8"))
PASSWORD_ITERATIONS = int(os.environ.get("AIOPS_PASSWORD_ITERATIONS", "260000"))
PASSWORD_MIN_LENGTH = int(os.environ.get("AIOPS_PASSWORD_MIN_LENGTH", "10"))
SESSION_COOKIE_SECURE = os.environ.get("AIOPS_SESSION_COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes", "on"}
SESSION_COOKIE_SAMESITE = os.environ.get("AIOPS_SESSION_COOKIE_SAMESITE", "lax").strip().lower() or "lax"
SESSION_BIND_USER_AGENT = os.environ.get("AIOPS_SESSION_BIND_USER_AGENT", "true").strip().lower() in {"1", "true", "yes", "on"}
SESSION_BIND_CLIENT_HOST = os.environ.get("AIOPS_SESSION_BIND_CLIENT_HOST", "").strip().lower() in {"1", "true", "yes", "on"}

# ROLE_PERMISSIONS: dict[str, set[str]] = {
#     "admin": {
#         "read",
#         "live_analyze",
#         "recovery_execute",
#         "feedback_write",
#         "model_select",
#         "model_promote",
#         "audit_view",
#         "user_manage",
#     },
#     "operator": {
#         "read",
#         "live_analyze",
#         "recovery_execute",
#         "feedback_write",
#     },
#     "viewer": {
#         "read",
#     },
#     "ml_engineer": {
#         "read",
#         "feedback_write",
#         "model_select",
#         "model_promote",
#         "audit_view",
#     },
# }

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {
        "read",
        "demo_analyze",
        "live_analyze",
        "recovery_execute",
        "feedback_write",
        "model_select",
        "model_promote",
        "audit_view",
        "user_manage",
    },
    "operator": {
        "read",
        "demo_analyze",
        "live_analyze",
        "recovery_execute",
    },
    "viewer": {
        "read",
    },
    "ml_engineer": {
        "read",
        "demo_analyze",
        "feedback_write",
        "model_select",
        "model_promote",
        "audit_view",
    },
}



@dataclass(frozen=True)
class AuthContext:
    username: str
    role: str
    session_id: str | None = None

    @property
    def user(self) -> str:
        return self.username

    @property
    def permissions(self) -> set[str]:
        return ROLE_PERMISSIONS.get(self.role, set())


def _client_host(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _password_policy_ok(password: str) -> bool:
    if len(password) < PASSWORD_MIN_LENGTH:
        return False
    classes = 0
    classes += any(ch.islower() for ch in password)
    classes += any(ch.isupper() for ch in password)
    classes += any(ch.isdigit() for ch in password)
    classes += any(not ch.isalnum() for ch in password)
    return classes >= 3


def hash_password(password: str) -> str:
    if not _password_policy_ok(password):
        raise ValueError(
            f"Password must be at least {PASSWORD_MIN_LENGTH} characters and include 3 character classes."
        )
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_raw, salt_b64, digest_b64 = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
    except Exception:
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def require_valid_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized not in VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role: {role}. Supported roles: {sorted(VALID_ROLES)}",
        )
    return normalized


def set_session_cookie(response: JSONResponse, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_TTL_HOURS * 3600,
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
        secure=SESSION_COOKIE_SECURE,
        path="/",
    )


def clear_session_cookie(response: JSONResponse) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def make_session_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)).isoformat()


def bootstrap_admin_if_configured() -> dict[str, str] | None:
    if not AUTH_ENABLED:
        return None
    if count_users() > 0:
        return None

    username = os.environ.get("AIOPS_BOOTSTRAP_ADMIN_USERNAME", "").strip()
    password = os.environ.get("AIOPS_BOOTSTRAP_ADMIN_PASSWORD", "").strip()
    display_name = os.environ.get("AIOPS_BOOTSTRAP_ADMIN_DISPLAY_NAME", "Administrator").strip()
    if not username or not password:
        return None

    existing = get_user(username, include_password_hash=False)
    if existing is not None:
        return {"username": username, "status": "exists"}

    create_user(
        username=username,
        password_hash=hash_password(password),
        role="admin",
        display_name=display_name,
        is_active=True,
    )
    write_audit_log(
        action="auth.bootstrap_admin",
        target_type="user",
        target_id=username,
        actor="system",
        payload={"display_name": display_name},
    )
    return {"username": username, "status": "created"}


def get_auth_context(request: Request) -> AuthContext:
    if not AUTH_ENABLED:
        return AuthContext(username="anonymous", role="admin", session_id=None)

    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = get_active_session(
        session_id,
        user_agent=request.headers.get("user-agent", ""),
        client_host=_client_host(request),
        bind_user_agent=SESSION_BIND_USER_AGENT,
        bind_client_host=SESSION_BIND_CLIENT_HOST,
    )
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    return AuthContext(
        username=session["username"],
        role=session["role"],
        session_id=session_id,
    )


def require_permission(permission: str) -> Callable:
    def dependency(
        request: Request,
        auth: AuthContext = Depends(get_auth_context),
    ) -> AuthContext:
        if permission not in auth.permissions:
            write_audit_log(
                action="auth.denied",
                target_type="api",
                target_id=str(request.url.path),
                actor=auth.username,
                payload={
                    "role": auth.role,
                    "required_permission": permission,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Permission denied",
                    "user": auth.username,
                    "role": auth.role,
                    "required_permission": permission,
                },
            )
        return auth

    return dependency

