"""Run the agent over the golden set and report the three-level metrics.

Needs a live model endpoint (see scripts/serve_vllm.sh / serve_ollama.sh).

Run:  python -m eval.run_eval
      python -m eval.run_eval --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from audience_agent import db
from audience_agent.agent import run_agent
from eval.metrics import aggregate, compare, PairMetrics

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden_set.yaml"
REPORTS = ROOT / "eval" / "reports"


def evaluate(limit: int = None) -> dict:
    with open(GOLDEN, encoding="utf-8") as f:
        items = yaml.safe_load(f)["items"]
    if limit:
        items = items[:limit]

    pairs, rows = [], []
    total_iters = 0
    for it in items:
        gold = db.audience_set(it["reference_sql"])
        t0 = time.time()
        try:
            res = run_agent(it["question"])
            total_iters += res.iterations
            pred = db.audience_set(res.audience_sql) if res.audience_sql else None
            err = None if res.audience_sql else "no audience query produced"
        except Exception as e:
            pred, err, res = None, str(e), None

        c = compare(gold, pred)
        ok = pred is not None
        pairs.append(PairMetrics(
            id=it["id"], gold_size=len(gold), pred_size=c["pred_size"], exact=c["exact"],
            jaccard=c["jaccard"], precision=c["precision"], recall=c["recall"],
            size_abs_err=c["size_abs_err"], ok=ok, error=err))
        rows.append({
            "id": it["id"], "question": it["question"], "gold_size": len(gold),
            "pred_size": c["pred_size"], "exact": c["exact"], "jaccard": round(c["jaccard"], 3),
            "iters": getattr(res, "iterations", None), "sql": getattr(res, "audience_sql", None),
            "sec": round(time.time() - t0, 1), "error": err})
        mark = "OK " if c["exact"] else ("~~ " if ok else "XX ")
        print(f"{mark}{it['id']}: gold={len(gold):>5d} pred={str(c['pred_size']):>5} "
              f"jac={c['jaccard']:.2f} {it['question'][:48]}")

    agg = aggregate(pairs)
    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k:16s} {v}")
    return {"aggregate": agg, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    report = evaluate(args.limit)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else REPORTS / "latest.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
