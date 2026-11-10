"""What the home page shows: the user's projects at a glance, in a handful
of queries however many projects there are."""
from sqlalchemy import func, select

from logic import projects
from utils import db

ATTENTION = ("alert", "warning")


def summary(user, runs=8):
    """Across every project the user can see: the projects, how many model
    versions are in production, approvals waiting, monitors that need a
    look, and the latest runs."""
    mine = projects.list_projects(user)
    names = {p["id"]: p["name"] for p in mine}
    out = {"projects": mine, "production": 0, "approvals": [],
           "monitors": [], "runs": []}
    if not names:
        return out
    ids = list(names)
    with db.session() as s:
        out["production"] = int(s.scalar(
            select(func.count()).select_from(db.ModelVersion).where(
                db.ModelVersion.project_id.in_(ids),
                db.ModelVersion.stage == "production")) or 0)
        waiting = s.execute(
            select(db.Transition, db.ModelVersion).join(
                db.ModelVersion,
                db.ModelVersion.id == db.Transition.version_id).where(
                db.Transition.project_id.in_(ids),
                db.Transition.status == "pending").order_by(
                db.Transition.requested_at.desc()))
        out["approvals"] = [
            {"project_id": t.project_id, "project": names[t.project_id],
             "model": v.name, "version": v.version, "to_stage": t.to_stage,
             "requested_by": t.requested_by, "requested_at": t.requested_at}
            for t, v in waiting]
        flagged = s.scalars(select(db.Monitor).where(
            db.Monitor.project_id.in_(ids), db.Monitor.enabled.is_(True),
            db.Monitor.last_status.in_(ATTENTION)).order_by(
            db.Monitor.last_run_at.desc()))
        out["monitors"] = [
            {"project_id": m.project_id, "project": names[m.project_id],
             "name": m.name, "model": m.model_name, "status": m.last_status,
             "last_run_at": m.last_run_at} for m in flagged]
        latest = s.scalars(select(db.Run).where(
            db.Run.project_id.in_(ids)).order_by(
            db.Run.created_at.desc()).limit(runs))
        out["runs"] = [
            {"project_id": r.project_id, "project": names[r.project_id],
             "task": r.task, "target": r.target, "best_model": r.best_model,
             "metric": r.metric, "score": r.score,
             "created_by": r.created_by, "created_at": r.created_at}
            for r in latest]
    return out
