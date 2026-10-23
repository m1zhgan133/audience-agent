"""End-to-end smoke test of the agent loop using a scripted fake LLM.

Verifies plumbing (prompt -> sql block -> guard -> read-only exec -> result feedback
-> audience capture -> FINAL) WITHOUT needing a served model. Exercises the happy
path, guard rejection + self-correction, and DuckDB-error + self-correction.

Run:  python scripts/smoke_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audience_agent.agent import run_agent


def scripted(responses):
    """Return an LLM fn that yields canned responses in order."""
    box = {"i": 0}

    def fn(_messages):
        r = responses[min(box["i"], len(responses) - 1)]
        box["i"] += 1
        return r

    return fn


def sql(q):
    return f"```sql\n{q}\n```"


def main() -> int:
    fails = 0

    def expect(cond, msg):
        nonlocal fails
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails += 1

    # 1. happy path
    r = run_agent("Клиенты с кредиткой", scripted([
        sql("SELECT client_id FROM dm_client_profile WHERE has_credit_card"),
        "FINAL ANSWER: 2146 клиентов с кредитной картой.",
    ]))
    expect(r.audience_size == 2146 and r.iterations == 2, f"happy path size={r.audience_size} iters={r.iterations}")

    # 2. guard rejection then correction
    r = run_agent("Удали клиентов", scripted([
        sql("DELETE FROM dm_client_profile"),
        sql("SELECT client_id FROM dm_client_profile WHERE segment='private'"),
        "FINAL ANSWER: 123 клиента.",
    ]))
    expect(r.audience_size == 123 and r.iterations == 3, f"guard->correct size={r.audience_size} iters={r.iterations}")

    # 3. DuckDB error then correction
    r = run_agent("Москвичи", scripted([
        sql("SELECT client_id FROM dm_client_profile WHERE region = 'MOW'"),  # wrong column
        sql("SELECT client_id FROM dm_client_profile WHERE region_code = 'MOW'"),
        "FINAL ANSWER: готово.",
    ]))
    ok = r.audience_size and r.audience_size > 0 and r.iterations == 3
    expect(ok, f"sqlerr->correct size={r.audience_size} iters={r.iterations}")

    print("\nSMOKE:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
