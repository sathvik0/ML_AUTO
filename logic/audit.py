"""The audit trail: who did what, when, to which project — tamper-evident.

Every event is written in the same database transaction as the action it
records (callers pass their session and commit), so an action cannot happen
without its event and a failed action leaves none.

Each event's hash is SHA-256 over the previous event's hash and the event's
own content, so changing or deleting any event breaks the chain from there
on; `verify()` walks it. `prev_hash` is unique, so two writers can never
fork the chain (the second fails instead); on PostgreSQL an advisory lock
serialises writers in the first place, and a trigger refuses UPDATE and
DELETE on the table outright.

Events carry ids, counts and settings — never data, credentials or the text
of prompts and queries. Each is also logged (`audit` events) for a SIEM.

Reading: admins see everything; project owners see their projects' events.
"""
import hashlib
import json
from datetime import timezone

from sqlalchemy import select, text

from utils import db, log

_log = log.get("audit")

GENESIS = "0" * 64
LOCK_ID = 7_310_456


def _actor(who):
    """(email, role) for a user dict, or a system name ("system:batch")."""
    if isinstance(who, str):
        return who, "system"
    email = (who.get("email") or "").strip().lower()
    return email or "unknown", who.get("role", "") or ""


def _content(at, actor, role, project_id, action, target_type, target_id,
             details):
    stamp = (at if at.tzinfo else at.replace(tzinfo=timezone.utc)) \
        .astimezone(timezone.utc).isoformat(timespec="microseconds")
    return json.dumps({"at": stamp, "actor": actor, "role": role,
                       "project": project_id, "action": action,
                       "type": target_type, "target": target_id,
                       "details": details}, sort_keys=True,
                      separators=(",", ":"))


def _hash(prev, content):
    return hashlib.sha256((prev + content).encode()).hexdigest()


def record(s, who, action, project_id=None, target_type="", target_id="",
           **details):
    """Add an event to the caller's session (it commits with the action).
    `details` must be small, non-sensitive facts: ids, counts, settings."""
    actor, role = _actor(who)
    details = json.loads(json.dumps(details, default=str))
    if s.bind.dialect.name == "postgresql":
        s.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": LOCK_ID})
    last = s.scalars(select(db.AuditEvent.hash)
                     .order_by(db.AuditEvent.seq.desc()).limit(1)).first()
    prev = last or GENESIS
    at = db.now()
    content = _content(at, actor, role, project_id, action, target_type,
                       str(target_id or ""), details)
    s.add(db.AuditEvent(at=at, actor=actor, actor_role=role,
                        project_id=project_id, action=action,
                        target_type=target_type,
                        target_id=str(target_id or ""), details=details,
                        prev_hash=prev, hash=_hash(prev, content)))
    s.flush()
    log.event(_log, "audit", action=action, actor_role=role,
              target_type=target_type)


def record_now(who, action, project_id=None, target_type="", target_id="",
               **details):
    """An event on its own (for actions with no transaction of their own)."""
    with db.session() as s:
        record(s, who, action, project_id, target_type, target_id,
               **details)
        s.commit()


def verify(batch=5000):
    """{"ok", "events", "first_bad"}: whether every event's hash matches
    its content and links to the one before."""
    prev, n, seq = GENESIS, 0, 0
    with db.session() as s:
        while True:
            rows = s.scalars(select(db.AuditEvent)
                             .where(db.AuditEvent.seq > seq)
                             .order_by(db.AuditEvent.seq).limit(batch)).all()
            if not rows:
                break
            for e in rows:
                content = _content(e.at, e.actor, e.actor_role, e.project_id,
                                   e.action, e.target_type, e.target_id,
                                   e.details)
                if e.prev_hash != prev or e.hash != _hash(prev, content):
                    return {"ok": False, "events": n, "first_bad": e.seq}
                prev, n, seq = e.hash, n + 1, e.seq
    return {"ok": True, "events": n, "first_bad": None}


def events(user, project_id=None, action=None, actor=None, limit=500):
    """The newest events the user may see: a project's for its owners,
    any for admins."""
    from logic.projects import AccessDenied, _need
    with db.session() as s:
        if project_id:
            _need(s, user, project_id, "owner")
        elif user.get("role") != "admin":
            raise AccessDenied("The full audit log is for admins.")
        q = select(db.AuditEvent)
        if project_id:
            q = q.where(db.AuditEvent.project_id == project_id)
        if action:
            q = q.where(db.AuditEvent.action.like(f"{action}%"))
        if actor:
            q = q.where(db.AuditEvent.actor == actor.strip().lower())
        rows = s.scalars(q.order_by(db.AuditEvent.seq.desc())
                         .limit(limit)).all()
        return [{"seq": e.seq, "at": e.at, "actor": e.actor,
                 "role": e.actor_role, "project_id": e.project_id,
                 "action": e.action, "target_type": e.target_type,
                 "target_id": e.target_id, "details": e.details,
                 "hash": e.hash} for e in rows]
