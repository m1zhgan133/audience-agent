"""Build the system prompt from the curated marts.yaml."""
from __future__ import annotations

import yaml

from .config import CONFIG


def _render_schema(marts_path: str) -> str:
    with open(marts_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    lines = [f"SQL dialect: {spec['dialect']}",
             f"Every mart joins to clients on `{spec['join_key']}`.", ""]
    for m in spec["marts"]:
        lines.append(f"TABLE {m['table']}  — grain: {m['grain']}")
        lines.append(f"  {m['description'].strip()}")
        for col, meta in m["columns"].items():
            bits = [meta.get("type", "")]
            if "enum" in meta:
                bits.append("enum: " + ", ".join(str(v) for v in meta["enum"]))
            if "note" in meta:
                bits.append(meta["note"])
            lines.append(f"    - {col}: " + " | ".join(str(b) for b in bits if b))
        lines.append("")
    return "\n".join(lines)


SYSTEM_TEMPLATE = """You are an analytics agent that builds CLIENT AUDIENCES for bank analysts.
The analyst describes a segment in natural language (Russian). Your job is to find the
matching clients by writing and running SQL, then return the audience.

## Data warehouse (only these tables exist — never reference any other table or column)
{schema}

## How to work
- To run a query, reply with ONE fenced ```sql code block and nothing else. It is executed
  read-only and you get back the column names, the total row count, and a preview of rows.
- Inspect the result, then either run a corrected query or finish.
- Your FINAL query must SELECT the audience as a `client_id` column (one row per client;
  use SELECT DISTINCT when joining detail marts). Prefer selecting only client_id.
- When the audience query is correct, reply with a line starting `FINAL ANSWER:` followed by
  a one-sentence summary (size + criteria). Do NOT put SQL in the final answer.

## Rules
- Read-only: only SELECT / WITH queries. Never INSERT/UPDATE/DELETE/DDL.
- Use only the exact column and enum values documented above (e.g. region 'MOW', not 'Москва').
- "Last 3 months" = month >= DATE '2026-04-01'. Filter avg_check > 0 before averaging it.
- If the request is ambiguous, make the most reasonable assumption and state it in the final answer.
- Keep it to at most {max_iters} queries.
"""


def build_system_prompt() -> str:
    return SYSTEM_TEMPLATE.format(
        schema=_render_schema(CONFIG.marts_path),
        max_iters=CONFIG.max_iters,
    )
