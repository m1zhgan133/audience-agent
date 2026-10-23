"""Audience-comparison metrics.

Three levels, coarse -> fine (mirrors the reference project, plus the composition
metric they said was "hard and expensive" and never shipped):
  1. exact_match        — the audience sets are identical
  2. size error (MAE/MSE aggregate) — |predicted| vs |gold|
  3. Jaccard / precision / recall — are they the *same* clients, not just the same count
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PairMetrics:
    id: str
    gold_size: int
    pred_size: Optional[int]
    exact: bool
    jaccard: float
    precision: float
    recall: float
    size_abs_err: Optional[int]
    ok: bool                 # did the agent produce a runnable audience at all
    error: Optional[str] = None


def compare(gold: set, pred: Optional[set]) -> dict:
    if pred is None:
        return {"exact": False, "jaccard": 0.0, "precision": 0.0, "recall": 0.0,
                "pred_size": None, "size_abs_err": None}
    inter = len(gold & pred)
    union = len(gold | pred)
    jac = inter / union if union else 1.0
    prec = inter / len(pred) if pred else (1.0 if not gold else 0.0)
    rec = inter / len(gold) if gold else (1.0 if not pred else 0.0)
    return {
        "exact": gold == pred,
        "jaccard": jac,
        "precision": prec,
        "recall": rec,
        "pred_size": len(pred),
        "size_abs_err": abs(len(pred) - len(gold)),
    }


def aggregate(pairs: List[PairMetrics]) -> dict:
    n = len(pairs)
    scored = [p for p in pairs if p.ok]
    errs = [p.size_abs_err for p in scored if p.size_abs_err is not None]
    mae = sum(errs) / len(errs) if errs else None
    mse = sum(e * e for e in errs) / len(errs) if errs else None
    return {
        "n": n,
        "runnable": len(scored),
        "crash_rate": round(1 - len(scored) / n, 4) if n else 0.0,
        "exact_match": round(sum(p.exact for p in pairs) / n, 4) if n else 0.0,
        "mean_jaccard": round(sum(p.jaccard for p in pairs) / n, 4) if n else 0.0,
        "mean_precision": round(sum(p.precision for p in pairs) / n, 4) if n else 0.0,
        "mean_recall": round(sum(p.recall for p in pairs) / n, 4) if n else 0.0,
        "size_mae": round(mae, 2) if mae is not None else None,
        "size_mse": round(mse, 2) if mse is not None else None,
    }
