"""Alembic environment: the app's models are the target, and the database is
the one `utils.db.url()` names (or the URL `migrate()` passes in).

On PostgreSQL an advisory lock is held for the duration, so several tasks
starting together migrate one at a time instead of racing."""
from alembic import context
from sqlalchemy import create_engine, text

from utils import db

LOCK_ID = 7_310_455          # any constant shared by every task

config = context.config
target_metadata = db.Base.metadata


def _url():
    return config.attributes.get("database_url") or db.url()


def run_offline():
    context.configure(url=_url(), target_metadata=target_metadata,
                      literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online():
    url = _url()
    engine = create_engine(url)
    with engine.connect() as conn:
        postgres = conn.dialect.name == "postgresql"
        if postgres:
            conn.execute(text("SELECT pg_advisory_lock(:id)"),
                         {"id": LOCK_ID})
        try:
            context.configure(connection=conn,
                              target_metadata=target_metadata,
                              # SQLite alters tables by copying them.
                              render_as_batch=not postgres)
            with context.begin_transaction():
                context.run_migrations()
        finally:
            if postgres:
                conn.execute(text("SELECT pg_advisory_unlock(:id)"),
                             {"id": LOCK_ID})
                conn.commit()
    engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
