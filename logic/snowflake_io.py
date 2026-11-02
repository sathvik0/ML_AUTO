"""Snowflake connectivity: read a table in, write the result set back.

A third vendor and a third credential set, so it lives in its own module with
its own resolution and its own failure handling — the app runs unchanged when
the package is absent — the panels simply say what is missing.

Nothing here raises. Every call returns `(value, error)` and every connection
or connector is injectable, which is how the suite exercises the SQL, the
identifier handling and the chunked write without an account.

Identifiers are always quoted. Snowflake folds an unquoted name to upper case,
so `Annual_Premium` written unquoted comes back as `ANNUAL_PREMIUM` and a
column named `Premium (norm)` is a syntax error. Quoting everything makes the
round trip exact — and because a quoted name must then be spelled exactly
forever, `normalize_columns` is offered first so the table lands with names a
SQL user can type.
"""
import functools
import logging
import re

import pandas as pd
from pandas.api.types import (is_bool_dtype, is_datetime64_any_dtype,
                              is_float_dtype, is_integer_dtype)

from utils import frames, log

_log = log.get("snowflake")


# A failure is logged as one of these words, never as its text: the
# connector's own message can quote a value it refused ("Numeric value 'x' is
# not recognized"). Matched against friendly_error's sentences; a sentence
# reworded without updating this falls to "other", which is the safe side.
_ERROR_KINDS = (
    ("Snowflake rejected those credentials", "credentials"),
    ("That database, schema or table does not exist", "not_found"),
    ("No warehouse is selected", "no_warehouse"),
    ("Could not reach that Snowflake account", "unreachable"),
    ("Snowflake needs an", "missing_settings"),
    ("Could not read that private key", "private_key"),
    ("There are no rows to write", "no_rows"),
    ("Write a query", "empty_query"),
    ("Only a query that reads data", "not_read_only"),
    ("Run one statement", "multiple_statements"),
    ("The query is longer", "too_long"),
    ("The query ran past its time limit", "timeout"),
)


def _error_kind(err) -> str:
    return next((kind for prefix, kind in _ERROR_KINDS
                 if str(err).startswith(prefix)), "other")


def _logged(name, size=None):
    """Log a `(value, error)` call's outcome: how long, whether it worked and
    how much it moved. Never its arguments — they carry the credentials and
    the rows being written. A failed write still reports what already landed:
    the difference between a retry and a duplicate."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with log.timer() as t:
                value, err = fn(*args, **kwargs)
            fields = {"ms": t.ms, "ok": err is None}
            if size is not None and value is not None:
                fields.update(size(value))
            if err:
                fields["error_kind"] = _error_kind(err)
            log.event(_log, name, level=logging.INFO if err is None
                      else logging.WARNING, **fields)
            return value, err
        return wrapper
    return deco

# write_pandas stages and COPYs; the fallback binds an INSERT per chunk. Both
# are chunked because a 300k-row single statement is a memory spike on the
# client and a single point of failure on the server.
CHUNK_ROWS = 50_000
# Seconds. Long enough for an SSO round trip, short enough that a wrong
# account id is an error rather than a frozen tab.
LOGIN_TIMEOUT = 30
MAX_IDENT = 255
# Snowflake's own limit is 16 MB per row; this is about the picker staying
# usable, not about the data.
MAX_PREVIEW_ROWS = 20
# A query typed into the app. Long enough for any real extract; a paste of a
# whole script is the thing this stops.
MAX_QUERY_CHARS = 20_000
# Seconds a read of the user's data may run. Without an explicit timeout the
# connector falls back to network_timeout (LOGIN_TIMEOUT, 30s) and cancels the
# statement — so a large table on a small warehouse failed at 30s. Long enough
# for a real extract; short enough that a runaway query does not hold the
# session, or the warehouse, for hours.
QUERY_TIMEOUT = 600

_ENV = {
    "account": "SNOWFLAKE_ACCOUNT",
    "user": "SNOWFLAKE_USER",
    "password": "SNOWFLAKE_PASSWORD",
    "role": "SNOWFLAKE_ROLE",
    "warehouse": "SNOWFLAKE_WAREHOUSE",
    "database": "SNOWFLAKE_DATABASE",
    "schema": "SNOWFLAKE_SCHEMA",
    "authenticator": "SNOWFLAKE_AUTHENTICATOR",
    "private_key_path": "SNOWFLAKE_PRIVATE_KEY_PATH",
    "private_key_passphrase": "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
}
FIELDS = tuple(_ENV)
# Kept out of session state once the connection is open, and never written to
# a file, a log line or an exported script.
SECRET_FIELDS = ("password", "private_key_passphrase")

WRITE_MODES = ("Create new table", "Replace table", "Append rows")


# ── availability ────────────────────────────────────────────────────────
def package_installed() -> bool:
    try:
        import snowflake.connector  # noqa: F401
        return True
    except ImportError:
        return False


def available() -> bool:
    """Whether the Snowflake panels can be offered at all."""
    return package_installed()


def unavailable_reason() -> str:
    return ("Snowflake needs the `snowflake-connector-python` package — "
            "`pip install \"snowflake-connector-python[pandas]\"`, then "
            "restart the app.")


def resolve_config() -> dict:
    """Env → st.secrets `[snowflake]` → the project's secrets.toml.

    Mirrors the Anthropic key resolution so a deployment can set whichever
    form it already uses. Everything is optional: what is missing is
    simply an empty box on the form.
    """
    import os
    out = {}
    for field, env in _ENV.items():
        value = os.environ.get(env)
        if value:
            out[field] = value
    try:
        import streamlit as st
        section = st.secrets.get("snowflake") or {}
        for field in FIELDS:
            if not out.get(field) and section.get(field):
                out[field] = str(section[field])
    except Exception:
        pass
    if len(out) < len(FIELDS):
        try:
            import os
            import tomllib
            root = os.path.dirname(os.path.dirname(__file__))
            with open(os.path.join(root, ".streamlit", "secrets.toml"),
                      "rb") as f:
                section = tomllib.load(f).get("snowflake") or {}
            for field in FIELDS:
                if not out.get(field) and section.get(field):
                    out[field] = str(section[field])
        except Exception:
            pass
    return out


def friendly_error(e) -> str:
    """Humanize the connector's errors; the raw ones name internal codes."""
    text = str(e)
    if "Incorrect username or password" in text or "250001" in text:
        return "Snowflake rejected those credentials."
    if "does not exist or not authorized" in text:
        return ("That database, schema or table does not exist, or this role "
                "cannot see it.")
    if "No active warehouse" in text:
        return ("No warehouse is selected for this session. Name one in the "
                "connection, or grant the role a default.")
    if "Could not connect" in text or "Name or service not known" in text:
        return "Could not reach that Snowflake account — check the account id."
    if "cancelled by the client due to a timeout" in text:
        return ("The query ran past its time limit and was stopped. A WHERE "
                "clause that narrows it, or a larger warehouse, finishes "
                "sooner.")
    return f"Snowflake: {text}"


# ── identifiers (pure) ──────────────────────────────────────────────────
def quote_ident(name) -> str:
    """A quoted identifier. Embedded quotes double, which is the escape."""
    return '"' + str(name).replace('"', '""') + '"'


def fqn(database, schema, table) -> str:
    """Fully-qualified, fully quoted: every part is spelled exactly."""
    parts = [p for p in (database, schema, table) if p]
    return ".".join(quote_ident(p) for p in parts)


def valid_ident(name):
    """(ok, message) for a name the user typed."""
    text = (name or "").strip()
    if not text:
        return False, "Name the table."
    if len(text) > MAX_IDENT:
        return False, f"A name is at most {MAX_IDENT} characters."
    if any(ord(c) < 32 for c in text):
        return False, "A name cannot contain control characters."
    return True, ""


_NON_SQL = re.compile(r"[^A-Za-z0-9_]+")


def normalize_columns(df: pd.DataFrame):
    """(renamed frame, {old: new}) with names a SQL user can type unquoted.

    `Premium (norm)` becomes `PREMIUM_NORM` and `Policy Type` becomes `POLICY_TYPE`.
    Collisions are numbered rather than dropped — two columns that normalize
    the same must still both land — and a name starting with a digit is
    prefixed, which Snowflake requires of an unquoted identifier.
    """
    renames, seen = {}, {}
    for col in df.columns:
        base = _NON_SQL.sub("_", str(col)).strip("_").upper() or "COLUMN"
        if base[0].isdigit():
            base = "C_" + base
        base = base[:MAX_IDENT]
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        renames[col] = base
    changed = {k: v for k, v in renames.items() if str(k) != v}
    return df.rename(columns=renames), changed


def sql_type(series) -> str:
    """The Snowflake column type for a pandas dtype.

    Deliberately coarse: NUMBER/FLOAT/BOOLEAN/TIMESTAMP_NTZ/VARCHAR covers
    every column this app produces, and VARCHAR in Snowflake is
    variable-length, so there is nothing to size.
    """
    if is_bool_dtype(series):
        return "BOOLEAN"
    if is_integer_dtype(series):
        return "NUMBER(38,0)"
    if is_float_dtype(series):
        return "FLOAT"
    if is_datetime64_any_dtype(series):
        return "TIMESTAMP_NTZ"
    return "VARCHAR"


def create_table_sql(df: pd.DataFrame, database, schema, table,
                     mode="Create new table") -> str:
    """The DDL for `df`, in the caller's chosen write mode.

    Written here rather than left to the connector's auto-create so the types
    are predictable and the statement can be shown to the user before it runs.
    """
    cols = ",\n    ".join(f"{quote_ident(c)} {sql_type(df[c])}"
                          for c in df.columns)
    target = fqn(database, schema, table)
    if mode == "Replace table":
        head = f"CREATE OR REPLACE TABLE {target}"
    elif mode == "Append rows":
        head = f"CREATE TABLE IF NOT EXISTS {target}"
    else:
        head = f"CREATE TABLE {target}"
    return f"{head} (\n    {cols}\n)"


def insert_sql(df: pd.DataFrame, database, schema, table) -> str:
    """Parameter-bound INSERT for the fallback path — values are never
    formatted into the statement, so nothing in the data can alter it."""
    cols = ", ".join(quote_ident(c) for c in df.columns)
    marks = ", ".join(["%s"] * len(df.columns))
    return (f"INSERT INTO {fqn(database, schema, table)} ({cols}) "
            f"VALUES ({marks})")


def chunks(df: pd.DataFrame, size=CHUNK_ROWS):
    """The frame in row blocks — one 300k-row statement is a spike at both
    ends of the wire."""
    size = max(1, int(size))
    for start in range(0, len(df), size):
        yield df.iloc[start:start + size]


# ── connecting ──────────────────────────────────────────────────────────
def _load_private_key(path, passphrase):
    try:
        from cryptography.hazmat.primitives import serialization
        with open(path, "rb") as f:
            key = serialization.load_pem_private_key(
                f.read(),
                password=(passphrase.encode() if passphrase else None))
        return key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()), None
    except FileNotFoundError:
        return None, f"No private key file at {path}."
    except Exception as e:
        return None, f"Could not read that private key: {e}"


@_logged("snowflake_connect")
def connect(params, connector=None):
    """(connection, error). `connector` is injectable for the smoke suite."""
    params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    for required in ("account", "user"):
        if not params.get(required):
            return None, f"Snowflake needs an {required}."
    if connector is None:
        if not package_installed():
            return None, unavailable_reason()
        import snowflake.connector as connector
    kwargs = {k: v for k, v in params.items() if k in FIELDS}
    key_path = kwargs.pop("private_key_path", None)
    passphrase = kwargs.pop("private_key_passphrase", None)
    if key_path:
        key, err = _load_private_key(key_path, passphrase)
        if err:
            return None, err
        kwargs["private_key"] = key
        kwargs.pop("password", None)     # key-pair auth, not both
    # A Streamlit rerun is single-threaded: a connect that hangs holds the
    # whole session, so it is bounded. Keep-alive stops a long editing session
    # from finding its token expired at the moment it writes.
    kwargs.setdefault("login_timeout", LOGIN_TIMEOUT)
    kwargs.setdefault("network_timeout", LOGIN_TIMEOUT)
    kwargs.setdefault("client_session_keep_alive", True)
    try:
        return connector.connect(**kwargs), None
    except Exception as e:
        return None, friendly_error(e)


def _rows(conn, sql, params=None):
    """Run a statement and return (rows, error). Cursors are always closed —
    a leaked cursor holds a result set open on the warehouse."""
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, params) if params else cur.execute(sql)
        return cur.fetchall(), None
    except Exception as e:
        return None, friendly_error(e)
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass


def probe(conn):
    """(identity dict, error) — the cheapest statement that proves a session."""
    rows, err = _rows(conn, "SELECT CURRENT_USER(), CURRENT_ROLE(), "
                            "CURRENT_WAREHOUSE(), CURRENT_DATABASE()")
    if err:
        return None, err
    row = list(rows[0]) if rows else ["", "", "", ""]
    keys = ("user", "role", "warehouse", "database")
    return dict(zip(keys, [("" if v is None else str(v)) for v in row])), None


def _names(conn, sql, column=1):
    rows, err = _rows(conn, sql)
    if err:
        return [], err
    return [str(r[column]) for r in rows], None


def list_databases(conn):
    return _names(conn, "SHOW DATABASES")


def list_schemas(conn, database):
    return _names(conn, f"SHOW SCHEMAS IN DATABASE {quote_ident(database)}")


def list_tables(conn, database, schema):
    """Tables and views — a dataset is as often a view."""
    target = f"{quote_ident(database)}.{quote_ident(schema)}"
    tables, err = _names(conn, f"SHOW TABLES IN SCHEMA {target}")
    if err:
        return [], err
    views, verr = _names(conn, f"SHOW VIEWS IN SCHEMA {target}")
    if verr:
        return sorted(tables), None      # tables alone are still usable
    return sorted(set(tables) | set(views)), None


# ── reading ─────────────────────────────────────────────────────────────
@_logged("snowflake_read",
         size=lambda df: {"rows": len(df), "columns": df.shape[1]})
def fetch_table(conn, database, schema, table, limit=0):
    """(frame, error). `limit=0` reads every row — the app analyses whole
    files, so a cap is only ever something the user asked for."""
    sql = f"SELECT * FROM {fqn(database, schema, table)}"
    if limit and int(limit) > 0:
        sql += f" LIMIT {int(limit)}"
    return _read(conn, sql)


def _read(conn, sql):
    """(frame, error) for one statement, every row it returns."""
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, timeout=QUERY_TIMEOUT)
        if hasattr(cur, "fetch_pandas_all"):
            # Arrow straight into pandas: the row-by-row path costs minutes on
            # a real extract.
            df = cur.fetch_pandas_all()
        else:
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
            df = pd.DataFrame(rows, columns=cols)
        # A query can hand back two columns of the same name; the schema keys
        # by name, so they are disambiguated here rather than colliding later.
        return frames.dedupe_columns(df), None
    except Exception as e:
        return None, friendly_error(e)
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass


# ── reading a query the user wrote ──────────────────────────────────────
def _scan(sql):
    """Walk SQL once, the way Snowflake reads it.

    Returns (code, bare, semicolons). `code` is the SQL with comments blanked
    out (quoted text kept); `bare` has quoted text blanked too, so a keyword
    test cannot be fooled by a string or an identifier; `semicolons` are the
    offsets of every ``;`` that ends a statement — one inside a string or a
    comment ends nothing. Blanking keeps every offset where it was.
    """
    code, bare, semis = [], [], []
    i, n = 0, len(sql)
    while i < n:
        two = sql[i:i + 2]
        if two in ("--", "//"):                  # both are line comments
            j = sql.find("\n", i)
            j = n if j < 0 else j
            code.append(" " * (j - i))
            bare.append(" " * (j - i))
        elif two == "/*":
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            code.append(" " * (j - i))
            bare.append(" " * (j - i))
        elif two == "$$":                        # dollar-quoted text
            j = sql.find("$$", i + 2)
            j = n if j < 0 else j + 2
            code.append(sql[i:j])
            bare.append(" " * (j - i))
        elif sql[i] in "'\"":
            quote, j = sql[i], i + 1
            while j < n:
                if quote == "'" and sql[j] == "\\":  # Snowflake's \' escape
                    j += 2
                elif sql[j] == quote and sql[j + 1:j + 2] == quote:
                    j += 2                       # a doubled quote
                elif sql[j] == quote:
                    break
                else:
                    j += 1
            j = min(j + 1, n)
            code.append(sql[i:j])
            bare.append(" " * (j - i))
        else:
            j = i + 1
            if sql[i] == ";":
                semis.append(i)
            code.append(sql[i])
            bare.append(sql[i])
        i = j
    return "".join(code), "".join(bare), semis


def check_query(sql):
    """(statement, error): the user's SQL, ready to run, or why it cannot.

    The box exists to *read* a slice of a large table, so it takes one
    statement that reads: SELECT, or WITH ... SELECT. Not a WITH that declares
    an anonymous procedure — Snowflake lets ``WITH p AS PROCEDURE ... CALL
    p()`` run arbitrary code behind a WITH. This guards against accidents, a
    pasted DELETE or a script; what the user's role may do is Snowflake's
    business, and a read-only role for the app is the real boundary.
    """
    text = (sql or "").strip()
    if not text:
        return None, "Write a query to run."
    if len(text) > MAX_QUERY_CHARS:
        return None, f"The query is longer than {MAX_QUERY_CHARS:,} characters."
    code, bare, semis = _scan(text)
    if semis:
        # One trailing semicolon is fine; anything after it but comments and
        # whitespace is a second statement.
        if len(semis) > 1 or code[semis[-1] + 1:].strip():
            return None, "Run one statement at a time."
        text = text[:semis[-1]].rstrip()
        bare = bare[:semis[-1]]
    # The whole first word, as Snowflake reads an unquoted name: `select1`
    # is one identifier, not SELECT followed by 1.
    first = re.match(r"[\s(]*([A-Za-z_][A-Za-z0-9_$]*)", bare)
    if (not first or first.group(1).upper() not in ("SELECT", "WITH")
            or re.search(r"\b(PROCEDURE|CALL)\b", bare, re.IGNORECASE)):
        return None, ("Only a query that reads data can run here: SELECT, "
                      "or WITH followed by SELECT.")
    return text, None


def _use_schema(conn, database, schema):
    """Resolve the query's unqualified names in the schema picked above it.

    USE changes the session's default only: everything else this module runs
    names its objects in full, so nothing else is affected."""
    if not (database and schema):
        return None
    _, err = _rows(conn, f"USE SCHEMA {quote_ident(database)}."
                         f"{quote_ident(schema)}")
    return err


@_logged("snowflake_preview",
         size=lambda got: {"rows": got[1], "columns": got[0].shape[1]})
def preview_query(conn, sql, database=None, schema=None,
                  rows=MAX_PREVIEW_ROWS):
    """((first rows, total rows), error) — what the query returns, before
    any of it is loaded.

    The query runs once in full; only `rows` of it cross the wire, and the
    total comes from the result's own row count. Snowflake normally serves
    the identical text from its result cache when Load runs it again, so the
    preview does not usually cost a second computation. The total is None
    when the driver does not report one.
    """
    stmt, err = check_query(sql)
    if err:
        return None, err
    err = _use_schema(conn, database, schema)
    if err:
        return None, err
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(stmt, timeout=QUERY_TIMEOUT)
        total = getattr(cur, "rowcount", None)
        head = list(cur.fetchmany(rows))
        cols = [d[0] for d in (cur.description or [])]
        frame = frames.dedupe_columns(pd.DataFrame(head, columns=cols))
        total = total if isinstance(total, int) and total >= 0 else None
        return (frame, total), None
    except Exception as e:
        return None, friendly_error(e)
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass


@_logged("snowflake_query",
         size=lambda df: {"rows": len(df), "columns": df.shape[1]})
def run_query(conn, sql, database=None, schema=None):
    """(frame, error) — every row the query returns.

    Logged by its size and timing only, never its text: a WHERE clause can
    name a person as easily as a category.
    """
    stmt, err = check_query(sql)
    if err:
        return None, err
    err = _use_schema(conn, database, schema)
    if err:
        return None, err
    return _read(conn, stmt)


# ── writing ─────────────────────────────────────────────────────────────
def _write_pandas(conn, df, database, schema, table):
    """Stage-and-COPY when the pandas extra is installed: it is the only path
    that is not row-by-row over the wire."""
    try:
        from snowflake.connector.pandas_tools import write_pandas
    except Exception:
        return None
    ok, _chunks, rows, _out = write_pandas(
        conn, df, table_name=table, database=database, schema=schema,
        quote_identifiers=True, chunk_size=CHUNK_ROWS)
    if not ok:
        raise RuntimeError("the staged load did not complete")
    return int(rows)


@_logged("snowflake_write", size=lambda n: {"rows": int(n)})
def write_table(conn, df: pd.DataFrame, database, schema, table,
                mode="Create new table", writer=None, chunk_rows=None):
    """Create (or replace, or reuse) the table and insert every row.

    Returns (rows written, error). The DDL is ours, so the column types are
    predictable and the statement could be shown before it ran; the rows go
    through the connector's staged loader when it is available and a
    parameter-bound INSERT per chunk when it is not.
    """
    ok, why = valid_ident(table)
    if not ok:
        return 0, why
    if df is None or df.empty:
        return 0, "There are no rows to write."
    if mode not in WRITE_MODES:
        return 0, f"Unknown write mode {mode!r}."

    # One statement per mode: CREATE (fails if it exists, which is the point
    # of "new"), CREATE OR REPLACE, or CREATE IF NOT EXISTS before an append.
    _, err = _rows(conn, create_table_sql(df, database, schema, table, mode))
    if err:
        return 0, err

    try:
        written = (writer or _write_pandas)(conn, df, database, schema, table)
    except Exception as e:
        return 0, friendly_error(e)
    if written is not None:
        return int(written), None

    # Fallback: bound INSERTs, chunked.
    sql = insert_sql(df, database, schema, table)
    total = 0
    for block in chunks(df, chunk_rows or CHUNK_ROWS):
        values = [tuple(None if pd.isna(v) else v for v in row)
                  for row in block.itertuples(index=False, name=None)]
        cur = None
        try:
            cur = conn.cursor()
            cur.executemany(sql, values)
            total += len(values)
        except Exception as e:
            return total, friendly_error(e)
        finally:
            if cur is not None:
                try:
                    cur.close()
                except Exception:
                    pass
    return total, None


def close(conn):
    """Best effort: a dropped session is not an error worth showing."""
    try:
        conn.close()
    except Exception:
        pass
