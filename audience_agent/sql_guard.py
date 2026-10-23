"""Defence-in-depth SQL validation.

The agent runs against a read-only DuckDB connection, so writes are already
impossible at the driver level. This guard rejects anything that isn't a single
read-only SELECT/WITH *before* it reaches the database, and gives the agent a
clear error message it can correct on.
"""
from __future__ import annotations

import re

# Whole-word keywords that must never appear in a read-only analytical query.
FORBIDDEN = [
    "insert", "update", "delete", "merge", "upsert",
    "drop", "alter", "create", "truncate", "replace",
    "attach", "detach", "copy", "export", "import",
    "install", "load", "pragma", "call", "vacuum",
    "set", "reset", "grant", "revoke",
]
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN) + r")\b", re.IGNORECASE)


class SQLGuardError(ValueError):
    """Raised when a query is not a safe single read-only statement."""


def guard(sql: str) -> str:
    """Return a cleaned, validated single-statement query or raise SQLGuardError."""
    if not sql or not sql.strip():
        raise SQLGuardError("empty query")

    # strip one trailing semicolon, then reject any further statement separators
    cleaned = sql.strip()
    cleaned = re.sub(r";\s*$", "", cleaned).strip()
    if ";" in cleaned:
        raise SQLGuardError("only a single statement is allowed (found ';')")

    first = cleaned.split(None, 1)[0].lower()
    if first not in ("select", "with"):
        raise SQLGuardError("only SELECT / WITH queries are allowed")

    hit = _FORBIDDEN_RE.search(cleaned)
    if hit:
        raise SQLGuardError(f"forbidden keyword '{hit.group(1).lower()}' — read-only queries only")

    return cleaned
