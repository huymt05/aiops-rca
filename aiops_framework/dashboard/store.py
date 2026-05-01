from __future__ import annotations

import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_DIR = Path(os.environ.get("AIOPS_DASHBOARD_STATE_DIR", "/tmp/aiops-dashboard"))
DB_PATH = Path(os.environ.get("AIOPS_DASHBOARD_DB_PATH", str(STATE_DIR / "aiops_dashboard.sqlite3")))
JSON_PATH = Path(os.environ.get("AIOPS_DASHBOARD_JSON_STORE_PATH", str(STATE_DIR / "aiops_dashboard_store.json")))
ALLOW_INSECURE_JSON_FALLBACK = os.environ.get("AIOPS_DASHBOARD_ALLOW_JSON_FALLBACK", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VALID_ROLES = {"admin", "operator", "viewer", "ml_engineer"}
_FALLBACK_TO_JSON = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_username(username: str) -> str:
    normalized = username.strip()
    if not normalized:
        raise ValueError("Username is required")
    return normalized


def _normalize_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role}. Supported roles: {sorted(VALID_ROLES)}")
    return normalized


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _empty_state() -> dict[str, Any]:
    return {
        "monitoring_events": [],
        "feedback": [],
        "audit_logs": [],
        "users": [],
        "user_sessions": [],
        "next_ids": {"monitoring_events": 1, "feedback": 1, "audit_logs": 1},
    }


def _read_state() -> dict[str, Any]:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not JSON_PATH.exists():
        return _empty_state()
    return json.loads(JSON_PATH.read_text(encoding="utf-8") or "{}") or _empty_state()


def _write_state(state: dict[str, Any]) -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _next_id(state: dict[str, Any], key: str) -> int:
    next_ids = state.setdefault("next_ids", {})
    value = int(next_ids.get(key, 1))
    next_ids[key] = value + 1
    return value


def init_store() -> None:
    global _FALLBACK_TO_JSON
    if _FALLBACK_TO_JSON:
        _write_state(_read_state())
        return
    try:
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS monitoring_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL,
                  system_id TEXT NOT NULL,
                  source_service TEXT NOT NULL,
                  status TEXT NOT NULL,
                  is_anomaly INTEGER NOT NULL,
                  anomaly_score REAL,
                  anomaly_model TEXT,
                  rca_top1_service TEXT,
                  rca_top1_score REAL,
                  recommendation_action TEXT,
                  payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_id INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  feedback TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  notes TEXT NOT NULL DEFAULT '',
                  payload_json TEXT NOT NULL,
                  FOREIGN KEY(event_id) REFERENCES monitoring_events(id)
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  action TEXT NOT NULL,
                  target_type TEXT NOT NULL,
                  target_id TEXT NOT NULL,
                  payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                  username TEXT PRIMARY KEY,
                  password_hash TEXT NOT NULL,
                  role TEXT NOT NULL,
                  display_name TEXT NOT NULL DEFAULT '',
                  is_active INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  last_login_at TEXT
                );

                CREATE TABLE IF NOT EXISTS user_sessions (
                  session_id TEXT PRIMARY KEY,
                  username TEXT NOT NULL,
                  role TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  revoked_at TEXT,
                  user_agent TEXT NOT NULL DEFAULT '',
                  client_host TEXT NOT NULL DEFAULT '',
                  FOREIGN KEY(username) REFERENCES users(username)
                );

                CREATE INDEX IF NOT EXISTS idx_user_sessions_username
                  ON user_sessions(username);

                CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at
                  ON user_sessions(expires_at);
                """
            )
    except sqlite3.Error as exc:
        if not ALLOW_INSECURE_JSON_FALLBACK:
            raise RuntimeError(
                "SQLite store initialization failed and insecure JSON fallback is disabled."
            ) from exc
        _FALLBACK_TO_JSON = True
        _write_state(_read_state())


def create_monitoring_event(result: dict[str, Any]) -> dict[str, Any]:
    init_store()
    live = result.get("live_context") or {}
    anomaly = result.get("anomaly") or {}
    rca = result.get("rca") or {}
    top1 = rca.get("top1") or {}
    recommendation = result.get("recommendation") or {}
    created_at = utc_now()
    system_id = str(live.get("system_id") or "unknown")
    source_service = str(live.get("source_service") or "unknown")
    is_anomaly = bool(anomaly.get("is_anomaly", False))
    status = "anomaly" if is_anomaly else "normal"
    if _FALLBACK_TO_JSON:
        state = _read_state()
        event_id = _next_id(state, "monitoring_events")
        state["monitoring_events"].append(
            {
                "id": event_id,
                "created_at": created_at,
                "system_id": system_id,
                "source_service": source_service,
                "status": status,
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly.get("anomaly_score"),
                "anomaly_model": anomaly.get("model_name"),
                "rca_top1_service": top1.get("service_name"),
                "rca_top1_score": top1.get("score"),
                "recommendation_action": recommendation.get("primary_action"),
                "payload": result,
            }
        )
        _write_state(state)
        return {
            "id": event_id,
            "created_at": created_at,
            "system_id": system_id,
            "source_service": source_service,
            "status": status,
            "is_anomaly": is_anomaly,
        }

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO monitoring_events (
              created_at, system_id, source_service, status, is_anomaly,
              anomaly_score, anomaly_model, rca_top1_service, rca_top1_score,
              recommendation_action, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                system_id,
                source_service,
                status,
                1 if is_anomaly else 0,
                anomaly.get("anomaly_score"),
                anomaly.get("model_name"),
                top1.get("service_name"),
                top1.get("score"),
                recommendation.get("primary_action"),
                _json_dump(result),
            ),
        )
        event_id = int(cursor.lastrowid)
    return {
        "id": event_id,
        "created_at": created_at,
        "system_id": system_id,
        "source_service": source_service,
        "status": status,
        "is_anomaly": is_anomaly,
    }


def _event_from_row(row: sqlite3.Row, include_payload: bool = False) -> dict[str, Any]:
    payload = {
        "id": int(row["id"]),
        "created_at": row["created_at"],
        "system_id": row["system_id"],
        "source_service": row["source_service"],
        "status": row["status"],
        "is_anomaly": bool(row["is_anomaly"]),
        "anomaly_score": row["anomaly_score"],
        "anomaly_model": row["anomaly_model"],
        "rca_top1_service": row["rca_top1_service"],
        "rca_top1_score": row["rca_top1_score"],
        "recommendation_action": row["recommendation_action"],
    }
    if include_payload:
        payload["payload"] = _json_load(row["payload_json"], {})
    return payload


def list_monitoring_events(limit: int = 50, system_id: str | None = None) -> list[dict[str, Any]]:
    init_store()
    limit = max(1, min(int(limit), 200))
    if _FALLBACK_TO_JSON:
        items = list(reversed(_read_state().get("monitoring_events", [])))
        if system_id:
            items = [item for item in items if item.get("system_id") == system_id]
        return [{key: value for key, value in item.items() if key != "payload"} for item in items[:limit]]
    with _connect() as conn:
        if system_id:
            rows = conn.execute(
                "SELECT * FROM monitoring_events WHERE system_id = ? ORDER BY id DESC LIMIT ?",
                (system_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM monitoring_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_event_from_row(row) for row in rows]


def get_monitoring_event(event_id: int) -> dict[str, Any] | None:
    init_store()
    if _FALLBACK_TO_JSON:
        for item in _read_state().get("monitoring_events", []):
            if int(item["id"]) == int(event_id):
                payload = {key: value for key, value in item.items() if key != "payload"}
                payload["payload"] = item.get("payload", {})
                return payload
        return None
    with _connect() as conn:
        row = conn.execute("SELECT * FROM monitoring_events WHERE id = ?", (int(event_id),)).fetchone()
    return _event_from_row(row, include_payload=True) if row else None


def write_audit_log(
    *,
    action: str,
    target_type: str,
    target_id: str,
    actor: str = "dashboard",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    init_store()
    created_at = utc_now()
    if _FALLBACK_TO_JSON:
        state = _read_state()
        audit_id = _next_id(state, "audit_logs")
        item = {
            "id": audit_id,
            "created_at": created_at,
            "actor": actor,
            "action": action,
            "target_type": target_type,
            "target_id": target_id,
            "payload": payload or {},
        }
        state["audit_logs"].append(item)
        _write_state(state)
        return {key: item[key] for key in ["id", "created_at", "actor", "action", "target_type", "target_id"]}
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO audit_logs (created_at, actor, action, target_type, target_id, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (created_at, actor, action, target_type, target_id, _json_dump(payload or {})),
        )
        audit_id = int(cursor.lastrowid)
    return {
        "id": audit_id,
        "created_at": created_at,
        "actor": actor,
        "action": action,
        "target_type": target_type,
        "target_id": target_id,
    }


def save_feedback(
    *,
    event_id: int,
    feedback: str,
    actor: str = "operator",
    notes: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    init_store()
    created_at = utc_now()
    if _FALLBACK_TO_JSON:
        state = _read_state()
        if not any(int(item["id"]) == int(event_id) for item in state.get("monitoring_events", [])):
            raise KeyError(f"Monitoring event not found: {event_id}")
        feedback_id = _next_id(state, "feedback")
        state["feedback"].append(
            {
                "id": feedback_id,
                "event_id": int(event_id),
                "created_at": created_at,
                "feedback": feedback,
                "actor": actor,
                "notes": notes,
                "payload": payload or {},
            }
        )
        _write_state(state)
        write_audit_log(
            action=f"feedback.{feedback}",
            target_type="monitoring_event",
            target_id=str(event_id),
            actor=actor,
            payload={"notes": notes, **(payload or {})},
        )
        return {
            "id": feedback_id,
            "event_id": int(event_id),
            "created_at": created_at,
            "feedback": feedback,
            "actor": actor,
            "notes": notes,
        }
    with _connect() as conn:
        event = conn.execute("SELECT id FROM monitoring_events WHERE id = ?", (int(event_id),)).fetchone()
        if event is None:
            raise KeyError(f"Monitoring event not found: {event_id}")
        cursor = conn.execute(
            """
            INSERT INTO feedback (event_id, created_at, feedback, actor, notes, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (int(event_id), created_at, feedback, actor, notes, _json_dump(payload or {})),
        )
        feedback_id = int(cursor.lastrowid)
    write_audit_log(
        action=f"feedback.{feedback}",
        target_type="monitoring_event",
        target_id=str(event_id),
        actor=actor,
        payload={"notes": notes, **(payload or {})},
    )
    return {
        "id": feedback_id,
        "event_id": int(event_id),
        "created_at": created_at,
        "feedback": feedback,
        "actor": actor,
        "notes": notes,
    }


def list_audit_logs(
    limit: int = 50,
    *,
    actor: str | None = None,
    action: str | None = None,
    system_id: str | None = None,
) -> list[dict[str, Any]]:
    init_store()
    limit = max(1, min(int(limit), 200))
    if _FALLBACK_TO_JSON:
        items = list(reversed(_read_state().get("audit_logs", [])))
        if actor:
            actor_filter = actor.strip().lower()
            items = [item for item in items if str(item.get("actor", "")).strip().lower() == actor_filter]
        if action:
            action_filter = action.strip().lower()
            items = [item for item in items if action_filter in str(item.get("action", "")).strip().lower()]
        if system_id:
            system_filter = system_id.strip().lower()
            items = [
                item for item in items
                if str((item.get("payload") or {}).get("system_id", "")).strip().lower() == system_filter
            ]
        return items[:limit]
    with _connect() as conn:
        clauses: list[str] = []
        params: list[Any] = []
        if actor:
            clauses.append("LOWER(actor) = ?")
            params.append(actor.strip().lower())
        if action:
            clauses.append("LOWER(action) LIKE ?")
            params.append(f"%{action.strip().lower()}%")

        query = "SELECT * FROM audit_logs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit if system_id is None else max(limit * 4, limit))
        rows = conn.execute(query, tuple(params)).fetchall()

    items = [
        {
            "id": int(row["id"]),
            "created_at": row["created_at"],
            "actor": row["actor"],
            "action": row["action"],
            "target_type": row["target_type"],
            "target_id": row["target_id"],
            "payload": _json_load(row["payload_json"], {}),
        }
        for row in rows
    ]
    if system_id:
        system_filter = system_id.strip().lower()
        items = [
            item for item in items
            if str((item.get("payload") or {}).get("system_id", "")).strip().lower() == system_filter
        ]
    return items[:limit]


def count_users() -> int:
    init_store()
    if _FALLBACK_TO_JSON:
        return len(_read_state().get("users", []))
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()
    return int(row["count"]) if row else 0


def create_user(
    *,
    username: str,
    password_hash: str,
    role: str,
    display_name: str = "",
    is_active: bool = True,
) -> dict[str, Any]:
    init_store()
    username = _normalize_username(username)
    role = _normalize_role(role)
    now = utc_now()

    if _FALLBACK_TO_JSON:
        state = _read_state()
        users = state.setdefault("users", [])
        if any(item.get("username") == username for item in users):
            raise sqlite3.IntegrityError(f"User already exists: {username}")
        item = {
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "display_name": display_name,
            "is_active": bool(is_active),
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        }
        users.append(item)
        _write_state(state)
        result = dict(item)
        result.pop("password_hash", None)
        return result

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (
              username, password_hash, role, display_name, is_active,
              created_at, updated_at, last_login_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (username, password_hash, role, display_name, 1 if is_active else 0, now, now),
        )

    user = get_user(username, include_password_hash=False)
    if user is None:
        raise KeyError(f"User not found after create: {username}")
    return user


def get_user(username: str, *, include_password_hash: bool = True) -> dict[str, Any] | None:
    init_store()
    username = _normalize_username(username)

    if _FALLBACK_TO_JSON:
        for item in _read_state().get("users", []):
            if item.get("username") == username:
                result = dict(item)
                if not include_password_hash:
                    result.pop("password_hash", None)
                return result
        return None

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT username, password_hash, role, display_name, is_active,
                   created_at, updated_at, last_login_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    if row is None:
        return None
    result = dict(row)
    result["is_active"] = bool(result["is_active"])
    if not include_password_hash:
        result.pop("password_hash", None)
    return result


def list_users() -> list[dict[str, Any]]:
    init_store()

    if _FALLBACK_TO_JSON:
        users = []
        for item in _read_state().get("users", []):
            result = dict(item)
            result.pop("password_hash", None)
            users.append(result)
        return sorted(users, key=lambda item: item.get("username", ""))

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT username, role, display_name, is_active,
                   created_at, updated_at, last_login_at
            FROM users
            ORDER BY username
            """
        ).fetchall()

    return [{**dict(row), "is_active": bool(row["is_active"])} for row in rows]


def update_user_role(username: str, role: str) -> dict[str, Any]:
    init_store()
    username = _normalize_username(username)
    role = _normalize_role(role)
    now = utc_now()

    if _FALLBACK_TO_JSON:
        state = _read_state()
        for item in state.get("users", []):
            if item.get("username") == username:
                item["role"] = role
                item["updated_at"] = now
                _write_state(state)
                revoke_user_sessions(username)
                result = dict(item)
                result.pop("password_hash", None)
                return result
        raise KeyError(f"User not found: {username}")

    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE users SET role = ?, updated_at = ? WHERE username = ?",
            (role, now, username),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"User not found: {username}")

    revoke_user_sessions(username)
    user = get_user(username, include_password_hash=False)
    if user is None:
        raise KeyError(f"User not found: {username}")
    return user


def set_user_active(username: str, is_active: bool) -> dict[str, Any]:
    init_store()
    username = _normalize_username(username)
    now = utc_now()

    if _FALLBACK_TO_JSON:
        state = _read_state()
        for item in state.get("users", []):
            if item.get("username") == username:
                item["is_active"] = bool(is_active)
                item["updated_at"] = now
                _write_state(state)
                revoke_user_sessions(username)
                result = dict(item)
                result.pop("password_hash", None)
                return result
        raise KeyError(f"User not found: {username}")

    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE users SET is_active = ?, updated_at = ? WHERE username = ?",
            (1 if is_active else 0, now, username),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"User not found: {username}")

    revoke_user_sessions(username)
    user = get_user(username, include_password_hash=False)
    if user is None:
        raise KeyError(f"User not found: {username}")
    return user


def update_user_password(username: str, password_hash: str) -> None:
    init_store()
    username = _normalize_username(username)
    now = utc_now()

    if _FALLBACK_TO_JSON:
        state = _read_state()
        for item in state.get("users", []):
            if item.get("username") == username:
                item["password_hash"] = password_hash
                item["updated_at"] = now
                _write_state(state)
                revoke_user_sessions(username)
                return
        raise KeyError(f"User not found: {username}")

    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE username = ?",
            (password_hash, now, username),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"User not found: {username}")

    revoke_user_sessions(username)


def cleanup_expired_sessions() -> int:
    init_store()
    now = utc_now()

    if _FALLBACK_TO_JSON:
        state = _read_state()
        kept: list[dict[str, Any]] = []
        removed = 0
        for item in state.get("user_sessions", []):
            if item.get("revoked_at") is None and _parse_timestamp(str(item.get("expires_at"))) is not None:
                expires_at = _parse_timestamp(str(item.get("expires_at")))
                if expires_at is not None and expires_at <= _parse_timestamp(now):
                    removed += 1
                    continue
            kept.append(item)
        state["user_sessions"] = kept
        _write_state(state)
        return removed

    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM user_sessions WHERE revoked_at IS NULL AND expires_at <= ?",
            (now,),
        )
    return int(cursor.rowcount or 0)


def create_session(
    *,
    username: str,
    expires_at: str,
    user_agent: str = "",
    client_host: str = "",
) -> str:
    init_store()
    cleanup_expired_sessions()
    username = _normalize_username(username)
    session_id = secrets.token_urlsafe(32)
    now = utc_now()
    normalized_user_agent = user_agent.strip()[:512]
    normalized_client_host = client_host.strip()[:128]

    if _FALLBACK_TO_JSON:
        state = _read_state()
        user = None
        for item in state.get("users", []):
            if item.get("username") == username:
                user = item
                break
        if user is None:
            raise KeyError(f"User not found: {username}")
        if not bool(user.get("is_active", False)):
            raise PermissionError(f"User is disabled: {username}")

        role = str(user.get("role", "")).strip().lower()
        state.setdefault("user_sessions", []).append(
            {
                "session_id": session_id,
                "username": username,
                "role": role,
                "created_at": now,
                "expires_at": expires_at,
                "revoked_at": None,
                "user_agent": normalized_user_agent,
                "client_host": normalized_client_host,
            }
        )
        user["last_login_at"] = now
        user["updated_at"] = now
        _write_state(state)
        return session_id

    with _connect() as conn:
        user = conn.execute(
            "SELECT username, role, is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if user is None:
            raise KeyError(f"User not found: {username}")
        if not bool(user["is_active"]):
            raise PermissionError(f"User is disabled: {username}")

        role = str(user["role"]).strip().lower()
        conn.execute(
            """
            INSERT INTO user_sessions (
              session_id, username, role, created_at, expires_at,
              user_agent, client_host
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, username, role, now, expires_at, normalized_user_agent, normalized_client_host),
        )
        conn.execute(
            "UPDATE users SET last_login_at = ?, updated_at = ? WHERE username = ?",
            (now, now, username),
        )
    return session_id


def get_active_session(
    session_id: str,
    *,
    user_agent: str = "",
    client_host: str = "",
    bind_user_agent: bool = False,
    bind_client_host: bool = False,
) -> dict[str, Any] | None:
    init_store()
    if not session_id:
        return None
    cleanup_expired_sessions()

    normalized_user_agent = user_agent.strip()[:512]
    normalized_client_host = client_host.strip()[:128]

    if _FALLBACK_TO_JSON:
        state = _read_state()
        session = None
        for item in state.get("user_sessions", []):
            expires_at = _parse_timestamp(str(item.get("expires_at")))
            if (
                item.get("session_id") == session_id
                and item.get("revoked_at") is None
                and expires_at is not None
                and expires_at > datetime.now(timezone.utc)
            ):
                session = dict(item)
                break
        if session is None:
            return None

        if bind_user_agent and session.get("user_agent", "") != normalized_user_agent:
            return None
        if bind_client_host and session.get("client_host", "") != normalized_client_host:
            return None

        user = None
        for item in state.get("users", []):
            if item.get("username") == session.get("username"):
                user = item
                break
        if user is None or not bool(user.get("is_active", False)):
            return None

        session["role"] = str(user.get("role", "")).strip().lower()
        return session

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
              s.session_id,
              s.username,
              u.role AS role,
              s.created_at,
              s.expires_at,
              s.revoked_at,
              s.user_agent,
              s.client_host
            FROM user_sessions s
            JOIN users u ON u.username = s.username
            WHERE s.session_id = ?
              AND s.revoked_at IS NULL
              AND s.expires_at > ?
              AND u.is_active = 1
            """,
            (session_id, utc_now()),
        ).fetchone()

    if row is None:
        return None
    session = dict(row)
    if bind_user_agent and session.get("user_agent", "") != normalized_user_agent:
        return None
    if bind_client_host and session.get("client_host", "") != normalized_client_host:
        return None
    return session


def revoke_session(session_id: str) -> None:
    init_store()
    if not session_id:
        return
    now = utc_now()

    if _FALLBACK_TO_JSON:
        state = _read_state()
        for item in state.get("user_sessions", []):
            if item.get("session_id") == session_id and item.get("revoked_at") is None:
                item["revoked_at"] = now
                break
        _write_state(state)
        return

    with _connect() as conn:
        conn.execute(
            "UPDATE user_sessions SET revoked_at = ? WHERE session_id = ?",
            (now, session_id),
        )


def revoke_user_sessions(username: str) -> int:
    init_store()
    username = _normalize_username(username)
    now = utc_now()

    if _FALLBACK_TO_JSON:
        state = _read_state()
        count = 0
        for item in state.get("user_sessions", []):
            if item.get("username") == username and item.get("revoked_at") is None:
                item["revoked_at"] = now
                count += 1
        _write_state(state)
        return count

    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE user_sessions
            SET revoked_at = ?
            WHERE username = ?
              AND revoked_at IS NULL
            """,
            (now, username),
        )
    return int(cursor.rowcount or 0)
