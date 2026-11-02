"""Snowflake panels: one connection, a source picker, and a writer.

The connection is the app's, not a page's — opened once and kept in session
state so the landing page and the export page share it. Everything else is a
thin shell over `logic.snowflake_io`, which owns the SQL.

The password is used to open the connection and then dropped: it is cleared
from the stored parameters, and the box itself is remounted under a new key so
the typed value stops living in session state too (a keyed widget's value
cannot be blanked in place — see components/input_keys.py for the same
mechanic and why).
"""
import pandas as pd
import streamlit as st

from logic import snowflake_io as sf

CONN = "sf_conn"          # the live connection
IDENT = "sf_identity"     # who Snowflake says we are
PARAMS = "sf_params"      # what opened it, minus the secrets
NONCE = "sf_form_nonce"   # steps after a connect, to remount the secret boxes

_AUTH = {"Password": "snowflake",
         "Key pair": "snowflake",
         "Browser SSO": "externalbrowser"}
SOURCES = ("Table", "Query")
# Above this a query's row count is called out before it loads: every row is
# held in memory (~0.5 GB per 300,000 rows, measured), on a server other
# sessions share.
LARGE_RESULT_ROWS = 1_000_000


def connected() -> bool:
    return st.session_state.get(CONN) is not None


def _nonce():
    return st.session_state.get(NONCE, 0)


def disconnect():
    conn = st.session_state.pop(CONN, None)
    if conn is not None:
        sf.close(conn)
    for key in (IDENT, PARAMS):
        st.session_state.pop(key, None)
    st.session_state[NONCE] = _nonce() + 1


def _connect(params):
    conn, err = sf.connect(params)
    if err:
        st.session_state.pop(CONN, None)
        return err
    identity, perr = sf.probe(conn)
    if perr:
        sf.close(conn)
        st.session_state.pop(CONN, None)
        return perr
    st.session_state[CONN] = conn
    st.session_state[IDENT] = identity
    # Kept so the pickers can default sensibly — never the secrets.
    st.session_state[PARAMS] = {k: v for k, v in params.items()
                                if k not in sf.SECRET_FIELDS}
    st.session_state[NONCE] = _nonce() + 1
    return None


def connection_form(key_prefix):
    """Render the connection state, or the form that opens one."""
    if not sf.available():
        st.info(sf.unavailable_reason())
        return False
    if connected():
        who = st.session_state.get(IDENT) or {}
        line = " · ".join(v for v in (who.get("user"), who.get("role"),
                                      who.get("warehouse")) if v)
        c1, c2 = st.columns([0.75, 0.25], vertical_alignment="center")
        c1.success(f"Connected to Snowflake — {line}", icon="❄️")
        if c2.button("Disconnect", key=f"{key_prefix}_sf_disconnect",
                     width='stretch'):
            disconnect()
            st.rerun()
        return True

    cfg = {**sf.resolve_config(), **(st.session_state.get(PARAMS) or {})}
    n = _nonce()
    with st.form(f"{key_prefix}_sf_form"):
        c1, c2, c3 = st.columns(3)
        account = c1.text_input("Account", value=cfg.get("account", ""),
                                placeholder="xy12345.us-east-1",
                                key=f"{key_prefix}_sf_account_{n}")
        user = c2.text_input("User", value=cfg.get("user", ""),
                             key=f"{key_prefix}_sf_user_{n}")
        method = c3.selectbox("Authentication", list(_AUTH),
                              key=f"{key_prefix}_sf_auth_{n}")
        secret, key_path, passphrase = "", "", ""
        if method == "Password":
            secret = st.text_input("Password", type="password",
                                   value=cfg.get("password", ""),
                                   key=f"{key_prefix}_sf_pwd_{n}")
        elif method == "Key pair":
            k1, k2 = st.columns(2)
            key_path = k1.text_input("Private key file",
                                     value=cfg.get("private_key_path", ""),
                                     key=f"{key_prefix}_sf_keypath_{n}")
            passphrase = k2.text_input("Key passphrase", type="password",
                                       key=f"{key_prefix}_sf_keypass_{n}")
        else:
            st.caption("Browser SSO opens an identity-provider tab on the "
                       "machine running the app. Use a password or a key pair "
                       "when the app is hosted elsewhere.")
        c4, c5, c6 = st.columns(3)
        role = c4.text_input("Role", value=cfg.get("role", ""),
                             key=f"{key_prefix}_sf_role_{n}")
        warehouse = c5.text_input("Warehouse", value=cfg.get("warehouse", ""),
                                  key=f"{key_prefix}_sf_wh_{n}")
        database = c6.text_input("Database", value=cfg.get("database", ""),
                                 key=f"{key_prefix}_sf_db_{n}")
        go = st.form_submit_button("Connect", type="primary")
    st.caption("Credentials are used to open the session and are not saved. "
               "Set them once in `.streamlit/secrets.toml` under "
               "`[snowflake]` to have the boxes filled in.")
    if go:
        params = {"account": account.strip(), "user": user.strip(),
                  "role": role.strip(), "warehouse": warehouse.strip(),
                  "database": database.strip(),
                  "authenticator": _AUTH[method]}
        if method == "Password":
            params["password"] = secret
        elif method == "Key pair":
            params["private_key_path"] = key_path.strip()
            params["private_key_passphrase"] = passphrase
        with st.spinner("Connecting to Snowflake"):
            err = _connect(params)
        if err:
            st.error(err)
        else:
            st.rerun()
    return connected()


def _picker(key_prefix, want_table=True):
    """Database → schema → table, each list read from the live session.

    Returns (database, schema, table). The lists come from SHOW, so a role
    that cannot see an object simply does not see it here either.
    """
    conn = st.session_state[CONN]
    params = st.session_state.get(PARAMS) or {}
    dbs, err = sf.list_databases(conn)
    if err:
        st.error(err)
        return None, None, None
    if not dbs:
        st.warning("This role can see no databases.")
        return None, None, None
    default_db = params.get("database")
    # One full-width row each, in smaller type (style.css). Three across the
    # upload page's middle column left ~190px a box, and a long name was cut
    # off with nothing on hover to read the rest; two across still cut a
    # 32-character database name (311px of text in a 244px box).
    database = st.selectbox(
        "Database", dbs, key=f"{key_prefix}_sf_pick_db",
        index=dbs.index(default_db) if default_db in dbs else 0)
    schemas, err = sf.list_schemas(conn, database)
    if err:
        st.error(err)
        return database, None, None
    schemas = [s for s in schemas if s != "INFORMATION_SCHEMA"] or schemas
    if not schemas:
        st.warning("That database has no schemas this role can see.")
        return database, None, None
    default_schema = params.get("schema")
    schema = st.selectbox(
        "Schema", schemas, key=f"{key_prefix}_sf_pick_schema",
        index=(schemas.index(default_schema)
               if default_schema in schemas else 0))
    if not want_table:
        return database, schema, None
    tables, err = sf.list_tables(conn, database, schema)
    if err:
        st.error(err)
        return database, schema, None
    if not tables:
        st.warning("That schema holds no tables or views this role can see.")
        return database, schema, None
    table = st.selectbox("Table or view", tables,
                         key=f"{key_prefix}_sf_pick_table")
    return database, schema, table


def source_picker(key_prefix="load"):
    """Pick a table and read it, or write a query and read what it returns.
    Returns a frame, or None until Load is pressed. A whole table by default —
    the app analyses every row; a query is how a very large table is cut down
    to the rows that matter before any of it crosses the wire."""
    if not connection_form(key_prefix):
        return None
    # No help= here or on the query box: the tooltip icon is a button, and
    # the app-wide pill rule paints it into a dark dot. Captions carry it.
    source = st.radio("Read from", SOURCES, horizontal=True,
                      key=f"{key_prefix}_sf_source")
    if source == "Query":
        return _query_source(key_prefix)
    database, schema, table = _picker(key_prefix)
    if not table:
        return None
    # The full name, in a line that wraps: a long one is cut off in the box.
    st.caption(f"Reading {database}.{schema}.{table}")
    c1, c2 = st.columns([0.35, 0.65], vertical_alignment="bottom")
    limit = c1.number_input("Row limit (0 reads every row)", min_value=0,
                            value=0, step=1000,
                            key=f"{key_prefix}_sf_limit")
    if not c2.button(f"Load {table}", type="primary", width='stretch',
                     key=f"{key_prefix}_sf_load"):
        return None
    with st.spinner(f"Reading {table} from Snowflake"):
        df, err = sf.fetch_table(st.session_state[CONN], database, schema,
                                 table, limit)
    if err:
        st.error(err)
        return None
    if df is None or df.empty:
        st.warning(f"{table} returned no rows.")
        return None
    return df


def _query_source(key_prefix):
    """A read-only query, previewed before it loads.

    Preview runs it and shows the total row count and the first rows, which is
    the point for a very large table: the size is known before anything is
    held in memory. Load is allowed without a preview.
    """
    database, schema, _ = _picker(key_prefix, want_table=False)
    if not schema:
        return None
    conn = st.session_state[CONN]
    sql = st.text_area(
        "Query", key=f"{key_prefix}_sf_sql", height=150,
        placeholder="SELECT *\nFROM CUSTOMERS\nWHERE REGION = 'West'")
    st.caption(f"One statement that reads data: SELECT, or WITH followed by "
               f"SELECT. Names without a database and schema resolve in "
               f"{database}.{schema}.")
    _, why = sf.check_query(sql)
    if sql.strip() and why:
        st.warning(why)

    pkey = f"{key_prefix}_sf_preview"
    preview = st.session_state.get(pkey)
    if preview and preview["for"] != (sql, database, schema):
        preview = None                  # the text or the context has moved on
    c1, c2 = st.columns(2)
    if c1.button("Preview", width='stretch', disabled=bool(why),
                 key=f"{key_prefix}_sf_preview_btn"):
        with st.spinner("Running the query"):
            got, err = sf.preview_query(conn, sql, database, schema)
        if err:
            st.error(err)
            preview = None
        else:
            preview = {"for": (sql, database, schema), "frame": got[0],
                       "total": got[1]}
        st.session_state[pkey] = preview
    total = preview["total"] if preview else None
    load = c2.button(f"Load {_count(total, 'row')}" if total
                     else "Load query results",
                     type="primary", width='stretch',
                     disabled=bool(why) or total == 0,
                     key=f"{key_prefix}_sf_query_load")
    if preview:
        head = preview["frame"]
        cols = _count(head.shape[1], "column")
        if total == 0:
            st.caption("The query returns no rows.")
        else:
            if total is None:
                st.caption(f"{cols}. The first {_count(len(head), 'row')} "
                           f"are below.")
            elif total <= len(head):
                st.caption(f"{_count(total, 'row')} × {cols}, all below.")
            else:
                st.caption(f"{_count(total, 'row')} × {cols}. The first "
                           f"{len(head):,} are below.")
            if total is not None and total > LARGE_RESULT_ROWS:
                st.warning(f"{total:,} rows. Every row is held in memory, "
                           f"about 0.5 GB per 300,000; a WHERE clause that "
                           f"narrows the query loads faster.")
            st.dataframe(head, hide_index=True, width='stretch')
    if not load:
        return None
    with st.spinner("Loading the query results"):
        df, err = sf.run_query(conn, sql, database, schema)
    if err:
        st.error(err)
        return None
    if df is None or df.empty:
        st.warning("The query returned no rows.")
        return None
    return df


def _count(n, noun):
    """'1 row', '1,478 rows'."""
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def writer(frame: pd.DataFrame, key_prefix="export"):
    """Write the rows in `frame` to a table the user names."""
    if not connection_form(key_prefix):
        return
    database, schema, _ = _picker(key_prefix, want_table=False)
    if not schema:
        return

    c1, c2 = st.columns([0.55, 0.45])
    table = c1.text_input("Table", value="ML_PREDICTIONS",
                          key=f"{key_prefix}_sf_table")
    mode = c2.selectbox("If it exists", sf.WRITE_MODES,
                        key=f"{key_prefix}_sf_mode",
                        help="Create new fails when the table is already "
                             "there. Replace drops and rebuilds it. Append "
                             "adds these rows to what is there.")
    ok, why = sf.valid_ident(table)
    if not ok:
        st.caption(why)
        return

    out, renamed = sf.normalize_columns(frame)
    if renamed:
        shown = ", ".join(f"{k} → {v}" for k, v in list(renamed.items())[:4])
        more = len(renamed) - 4
        st.caption(f"Columns are renamed for SQL: {shown}"
                   + (f" and {more} more" if more > 0 else ""))
    with st.expander("Statement that will run"):
        st.code(sf.create_table_sql(out, database, schema, table, mode),
                language="sql")

    st.caption(f"{len(out):,} rows × {out.shape[1]} columns → "
               f"{sf.fqn(database, schema, table)}")
    if not st.button(f"Write {len(out):,} rows", type="primary",
                     key=f"{key_prefix}_sf_write"):
        return
    with st.spinner("Writing to Snowflake"):
        rows, err = sf.write_table(st.session_state[CONN], out, database,
                                   schema, table, mode)
    if err:
        # Partial writes are real: an append that fails halfway has already
        # committed its earlier chunks, and saying so is the difference
        # between a retry and a duplicate.
        if rows:
            st.error(f"{err} {rows:,} rows were written before it stopped.")
        else:
            st.error(err)
        return
    st.success(f"Wrote {rows:,} rows to {sf.fqn(database, schema, table)}",
               icon="❄️")
