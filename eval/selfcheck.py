"""Validate the harness WITHOUT an LLM.

Checks that every golden-set reference query:
  - passes the SQL guard,
  - runs read-only and returns a client_id audience,
  - matches its documented expected_size,
  - is deterministic (same set on a second run),
and that the metric functions behave. Run this after generate_data.py to prove the
data + guard + metrics + golden set are all self-consistent.

Run:  python -m eval.selfcheck
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from audience_agent import db
from audience_agent.sql_guard import guard
from eval.metrics import aggregate, compare, PairMetrics

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden_set.yaml"


def check_golden() -> int:
    with open(GOLDEN, encoding="utf-8") as f:
        items = yaml.safe_load(f)["items"]
    failures = 0
    for it in items:
        gid, sql = it["id"], it["reference_sql"]
        try:
            guard(sql)
            s1 = db.audience_set(sql)
            s2 = db.audience_set(sql)
        except Exception as e:
            print(f"  FAIL {gid}: {e}")
            failures += 1
            continue
        problems = []
        if not s1:
            problems.append("empty audience")
        if s1 != s2:
            problems.append("non-deterministic")
        exp = it.get("expected_size")
        if exp is not None and len(s1) != exp:
            problems.append(f"size {len(s1)} != expected {exp}")
        if problems:
            print(f"  FAIL {gid}: {'; '.join(problems)}")
            failures += 1
        else:
            print(f"  ok   {gid}: {len(s1):>5d} clients")
    print(f"golden set: {len(items) - failures}/{len(items)} passed")
    return failures


def check_metrics() -> int:
    a = {1, 2, 3, 4}
    fails = 0

    def expect(cond, msg):
        nonlocal fails
        if not cond:
            print(f"  FAIL metric: {msg}")
            fails += 1

    m = compare(a, set(a))
    expect(m["exact"] and m["jaccard"] == 1.0 and m["recall"] == 1.0, "identical sets")
    m = compare(a, {1, 2})              # perfect precision, half recall
    expect(not m["exact"] and m["precision"] == 1.0 and m["recall"] == 0.5, "subset")
    m = compare(a, {1, 2, 3, 4, 5, 6})  # all found + extras
    expect(m["recall"] == 1.0 and abs(m["precision"] - 4 / 6) < 1e-9, "superset")
    m = compare(a, None)                # agent failed
    expect(m["jaccard"] == 0.0 and m["pred_size"] is None, "no prediction")

    agg = aggregate([
        PairMetrics("x", 4, 4, True, 1.0, 1.0, 1.0, 0, True),
        PairMetrics("y", 4, None, False, 0.0, 0.0, 0.0, None, False, "boom"),
    ])
    expect(agg["crash_rate"] == 0.5 and agg["exact_match"] == 0.5, "aggregate")
    print(f"metrics: {'all passed' if fails == 0 else str(fails) + ' failed'}")
    return fails


def main() -> int:
    print("== golden set ==")
    f1 = check_golden()
    print("== metrics ==")
    f2 = check_metrics()
    total = f1 + f2
    print("\nSELF-CHECK:", "PASS" if total == 0 else f"FAIL ({total} problems)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
