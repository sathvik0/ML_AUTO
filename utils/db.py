"""The app's database: projects, members, datasets, the saved workspace,
training runs and the model registry.

Where it lives:
  DATABASE_URL                      used as given, when set
  DB_HOST, DB_PORT, DB_USER,        PostgreSQL (Aurora on AWS: ECS injects
  DB_PASSWORD, DB_NAME              these from the cluster's own secret)
  otherwise                         SQLite in data/app.db, for local work

The schema is managed by Alembic (migrations/); `migrate()` brings the
database to the latest revision and runs once per process at start.
"""
import functools
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer,
                        String, Text, UniqueConstraint, create_engine)
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            sessionmaker)

ROOT = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc)


def new_id():
    return uuid.uuid4().hex


def url():
    env = os.environ
    if env.get("DATABASE_URL"):
        return env["DATABASE_URL"]
    if env.get("DB_HOST"):
        return (f"postgresql+psycopg://{quote_plus(env['DB_USER'])}:"
                f"{quote_plus(env['DB_PASSWORD'])}@{env['DB_HOST']}:"
                f"{env.get('DB_PORT', '5432')}/{env.get('DB_NAME', 'mlauto')}")
    path = ROOT / "data" / "app.db"
    path.parent.mkdir(exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    sub: Mapped[str] = mapped_column(String(128), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                default=now)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    owner_email: Mapped[str] = mapped_column(String(320), index=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    # Governance policy, e.g. {"require_fairness": true}.
    settings: Mapped[dict] = mapped_column(JSON, default=dict,
                                           server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)


class Member(Base):
    """Sharing is by email, so a colleague can be added before they first
    sign in."""
    __tablename__ = "members"
    __table_args__ = (UniqueConstraint("project_id", "email"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True,
                                    autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    role: Mapped[str] = mapped_column(String(16))      # owner/editor/viewer
    added_by: Mapped[str] = mapped_column(String(320), default="")
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                               default=now)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(32))  # upload/sample/snowflake
    rows: Mapped[int] = mapped_column(Integer)
    columns: Mapped[int] = mapped_column(Integer)
    object_key: Mapped[str] = mapped_column(String(500))
    created_by: Mapped[str] = mapped_column(String(320), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)


class Workspace(Base):
    """Where a project was left: its dataset, preparation steps and the
    configure page's choices."""
    __tablename__ = "workspaces"
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    steps: Mapped[list] = mapped_column(JSON, default=list)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_by: Mapped[str] = mapped_column(String(320), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    task: Mapped[str] = mapped_column(String(32))
    target: Mapped[str | None] = mapped_column(String(300), nullable=True)
    best_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metric: Mapped[str | None] = mapped_column(String(64), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    leaderboard: Mapped[list] = mapped_column(JSON, default=list)
    external: Mapped[bool] = mapped_column(Boolean, default=False)
    object_key: Mapped[str] = mapped_column(String(500))
    sklearn: Mapped[str] = mapped_column(String(32), default="")
    created_by: Mapped[str] = mapped_column(String(320), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)


class ModelVersion(Base):
    """The model registry: a named model's numbered versions, each with
    the run it came from and a stage (item 3 adds approvals)."""
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("project_id", "name", "version"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer)
    model_key: Mapped[str] = mapped_column(String(64))  # e.g. hist_gb
    object_key: Mapped[str] = mapped_column(String(500))
    task: Mapped[str] = mapped_column(String(32))
    target: Mapped[str | None] = mapped_column(String(300), nullable=True)
    metric: Mapped[str | None] = mapped_column(String(64), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage: Mapped[str] = mapped_column(String(16), default="none")
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(320), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)


class Transition(Base):
    """A request to move a model version to a stage. Production needs a
    second person's approval; the log of decisions is kept."""
    __tablename__ = "model_transitions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    to_stage: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))  # pending/approved/...
    comment: Mapped[str] = mapped_column(Text, default="")
    requested_by: Mapped[str] = mapped_column(String(320))
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                   default=now)
    decided_by: Mapped[str | None] = mapped_column(String(320),
                                                   nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    decision_note: Mapped[str] = mapped_column(Text, default="")


class ApiKey(Base):
    """A project's key for the model API. Only a hash is stored; the key
    itself is shown once, when it is made."""
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_by: Mapped[str] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)


class BatchJob(Base):
    """Scheduled scoring: a registered model's production (or staging)
    version applied to a source, written to a destination."""
    __tablename__ = "batch_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    model_name: Mapped[str] = mapped_column(String(200))
    stage: Mapped[str] = mapped_column(String(16), default="production")
    source: Mapped[dict] = mapped_column(JSON)
    destination: Mapped[dict] = mapped_column(JSON)
    schedule: Mapped[str] = mapped_column(String(100), default="")  # cron
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    next_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True)
    last_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    last_status: Mapped[str] = mapped_column(String(16), default="")
    created_by: Mapped[str] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)


class BatchRun(Base):
    __tablename__ = "batch_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("batch_jobs.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16))  # running/succeeded/...
    triggered_by: Mapped[str] = mapped_column(String(320))
    model_version_id: Mapped[str | None] = mapped_column(String(32),
                                                         nullable=True)
    rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_key: Mapped[str] = mapped_column(String(500), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)


class Monitor(Base):
    """A scheduled check of a model in production against newer data:
    drift, and accuracy where the data has actual outcomes."""
    __tablename__ = "monitors"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    model_name: Mapped[str] = mapped_column(String(200))
    source: Mapped[dict] = mapped_column(JSON)       # as a batch job's
    thresholds: Mapped[dict] = mapped_column(JSON)
    alert_webhook: Mapped[str] = mapped_column(String(500), default="")
    retrain: Mapped[str] = mapped_column(String(16), default="notify")
    schedule: Mapped[str] = mapped_column(String(100), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    next_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True)
    last_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    last_status: Mapped[str] = mapped_column(String(16), default="")
    created_by: Mapped[str] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)


class MonitorRun(Base):
    __tablename__ = "monitor_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    monitor_id: Mapped[str] = mapped_column(
        ForeignKey("monitors.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16))  # ok/warning/alert/...
    triggered_by: Mapped[str] = mapped_column(String(320))
    model_version_id: Mapped[str | None] = mapped_column(String(32),
                                                         nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    message: Mapped[str] = mapped_column(Text, default="")
    alerted: Mapped[bool] = mapped_column(Boolean, default=False)
    retrained_version_id: Mapped[str | None] = mapped_column(
        String(32), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)


class Job(Base):
    """Background work (training today): queued in the web app, run by a
    worker, watched and cancelled from the page."""
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16))           # train
    status: Mapped[str] = mapped_column(String(16), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    progress_text: Mapped[str] = mapped_column(String(300), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    worker: Mapped[str] = mapped_column(String(100), default="")
    created_by: Mapped[str] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)


class AuditEvent(Base):
    """One thing someone did. Append-only and hash-chained: each event's
    hash covers its content and the previous event's hash, so changing or
    removing any event breaks the chain from there on. The project is a
    plain id, not a foreign key: the history outlives a deleted project."""
    __tablename__ = "audit_events"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True,
                                     autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                         default=now, index=True)
    actor: Mapped[str] = mapped_column(String(320), index=True)
    actor_role: Mapped[str] = mapped_column(String(16), default="")
    project_id: Mapped[str | None] = mapped_column(String(32), nullable=True,
                                                   index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    target_type: Mapped[str] = mapped_column(String(32), default="")
    target_id: Mapped[str] = mapped_column(String(64), default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    # Unique: two events can never claim the same predecessor, so a race
    # fails loudly instead of forking the chain.
    prev_hash: Mapped[str] = mapped_column(String(64), unique=True)
    hash: Mapped[str] = mapped_column(String(64), unique=True)


class FairnessReport(Base):
    """How a model version treats groups of a sensitive attribute, on its
    out-of-sample predictions."""
    __tablename__ = "fairness_reports"
    id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                    default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    attributes: Mapped[list] = mapped_column(JSON)
    report: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))  # pass/flag/insufficient
    rows: Mapped[int] = mapped_column(Integer)
    created_by: Mapped[str] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=now)
    accepted_by: Mapped[str | None] = mapped_column(String(320),
                                                    nullable=True)
    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)
    acceptance_note: Mapped[str] = mapped_column(Text, default="")


def engine(database_url=None):
    return _engine(database_url or url())


@functools.lru_cache(maxsize=8)
def _engine(target):
    kwargs = {"pool_pre_ping": True}
    if target.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    elif os.environ.get("DB_POOL", "none") == "none":
        # No idle connections: each unit of work connects and closes, so
        # Aurora Serverless v2 can pause while nobody uses the app (an
        # always-on service holding a pool would keep it running). A few
        # milliseconds per session; DB_POOL=queue brings the pool back.
        kwargs = {"poolclass": NullPool}
    return create_engine(target, **kwargs)


def session(database_url=None):
    return sessionmaker(engine(database_url), expire_on_commit=False)()


def migrate(database_url=None):
    """The database at the latest Alembic revision. Once per process and
    database."""
    return _migrate(database_url or url())


@functools.lru_cache(maxsize=8)
def _migrate(target):
    from alembic import command
    from alembic.config import Config
    cfg = Config()
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["database_url"] = target
    command.upgrade(cfg, "head")
    return True
