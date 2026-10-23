"""Read-only DuckDB access used by the agent tool and the evaluation harness."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import duckdb

from .config import CONFIG
from .sql_guard import guard


def connect_ro() -> "duckdb.DuckDBPyConnection":
    return duckdb.connect(CONFIG.db_path, read_only=True)


def _wrap(sql: str) -> str:
    """Validate and strip trailing ';' so we can wrap the query as a subquery."""
    return guard(sql)


def run_preview(sql: str, preview_rows: int = None) -> Dict[str, Any]:
    """Execute a query read-only and return columns, a row preview and the total count.

    Raises whatever DuckDB raises on a bad query (the agent feeds the message back).
    """
    preview_rows = preview_rows if preview_rows is not None else CONFIG.preview_rows
    inner = _wrap(sql)
    con = connect_ro()
    try:
        total = con.execute(f"SELECT count(*) FROM ({inner}) _q").fetchone()[0]
        rel = con.execute(
            f"SELECT * FROM ({inner}) _q LIMIT {min(preview_rows, CONFIG.preview_hardcap)}"
        )
        cols: List[str] = [d[0] for d in rel.description]
        rows: List[Tuple] = rel.fetchall()
    finally:
        con.close()
    return {"columns": cols, "rows": rows, "total": int(total)}


def audience_set(sql: str) -> set:
    """Return the set of client_id values a query selects.

    The query must expose a `client_id` column; otherwise DuckDB raises and the
    caller treats it as a non-audience query.
    """
    inner = _wrap(sql)
    con = connect_ro()
    try:
        rows = con.execute(f"SELECT client_id FROM ({inner}) _q").fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


def has_client_id(sql: str) -> bool:
    inner = _wrap(sql)
    con = connect_ro()
    try:
        cols = [d[0] for d in con.execute(f"SELECT * FROM ({inner}) _q LIMIT 0").description]
    finally:
        con.close()
    return "client_id" in cols
