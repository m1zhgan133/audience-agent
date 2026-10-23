"""Generate the synthetic warehouse (10 marts, DuckDB), deterministically.

The marts match audience_agent/schema/marts.yaml exactly, and the profile has_*
flags are DERIVED from the detail marts, so flag filters and detail joins agree —
that's what makes golden-set answers well-defined.

The shipped data/warehouse.duckdb is canonical (golden expected_size values were
documented against it). Regenerating produces statistically similar but not
identical data, so either keep the shipped DB, or regenerate WITH --sync-golden
to rewrite expected_size values, then re-run: python -m eval.selfcheck.

Run:  python scripts/generate_data.py --out data/warehouse.duckdb --force [--sync-golden]
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N_CLIENTS = 5000
MONTHS = [date(2026, m, 1) for m in range(1, 7)]

REGIONS = {
    "MOW": ("Москва", 0.302), "SPB": ("Санкт-Петербург", 0.139),
    "MO": ("Подольск", 0.102), "EKB": ("Екатеринбург", 0.083),
    "NSK": ("Новосибирск", 0.069), "NN": ("Нижний Новгород", 0.069),
    "KDA": ("Краснодар", 0.066), "KZN": ("Казань", 0.065),
    "ROV": ("Ростов-на-Дону", 0.056), "LO": ("Гатчина", 0.049),
}
SEGMENTS = (["mass", "affluent", "private"], [0.80, 0.175, 0.025])


def _dates(rng, n, start=date(2015, 1, 1), end=date(2026, 6, 30)):
    span = (end - start).days
    return [start + pd.Timedelta(days=int(d)) for d in rng.integers(0, span, n)]


def generate(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cid = np.arange(1, N_CLIENTS + 1)

    region = rng.choice(list(REGIONS), N_CLIENTS, p=[w for _, w in REGIONS.values()])
    segment = rng.choice(SEGMENTS[0], N_CLIENTS, p=SEGMENTS[1])
    age = np.clip(rng.normal(38, 13, N_CLIENTS).astype(int), 18, 79)
    gender = rng.choice(["M", "F"], N_CLIENTS)
    reg_date = pd.to_datetime(_dates(rng, N_CLIENTS, end=date(2026, 5, 31)))
    tenure = ((pd.Timestamp("2026-06-30") - reg_date).days // 30).astype(int)

    seg_mult = np.where(segment == "private", 8.0, np.where(segment == "affluent", 2.5, 1.0))
    income = np.round(rng.lognormal(11.2, 0.55, N_CLIENTS) * seg_mult ** 0.5, -2)

    # --- product holdings (profile flags are derived from these rows) ---
    p_credit = np.clip(0.28 + 0.10 * (segment != "mass") + 0.15 * (income > 150_000), 0, 0.9)
    p_dep = np.clip(0.18 + 0.30 * (segment == "affluent") + 0.55 * (segment == "private")
                    + 0.08 * (age > 45), 0, 0.95)
    holds = {
        "debit_card": rng.random(N_CLIENTS) < 0.92,
        "credit_card": rng.random(N_CLIENTS) < p_credit,
        "deposit": rng.random(N_CLIENTS) < p_dep,
        "mortgage": rng.random(N_CLIENTS) < np.clip(0.06 + 0.10 * ((age > 27) & (age < 50)), 0, 1),
        "auto_loan": rng.random(N_CLIENTS) < 0.09,
        "cash_loan": rng.random(N_CLIENTS) < 0.16,
        "investment": rng.random(N_CLIENTS) < np.clip(0.05 + 0.25 * (segment != "mass"), 0, 1),
        "insurance": rng.random(N_CLIENTS) < 0.12,
    }
    prod_rows = []
    for p, mask in holds.items():
        ids = cid[mask]
        bal = np.round(rng.lognormal(11.5, 1.1, len(ids)) * seg_mult[mask] ** 0.5, 2)
        if p in ("credit_card", "mortgage", "auto_loan", "cash_loan"):
            bal = -bal
        prod_rows.append(pd.DataFrame({
            "client_id": ids, "product_type": p,
            "open_date": _dates(rng, len(ids)),
            "is_active": rng.random(len(ids)) < 0.93, "balance": bal}))
    products = pd.concat(prod_rows, ignore_index=True)

    profile = pd.DataFrame({
        "client_id": cid, "region_code": region,
        "city": [REGIONS[r][0] for r in region], "age": age, "gender": gender,
        "segment": segment, "registration_date": reg_date.date, "tenure_months": tenure,
        "is_active": rng.random(N_CLIENTS) < 0.86,
        "has_credit_card": holds["credit_card"], "has_debit_card": holds["debit_card"],
        "has_deposit": holds["deposit"], "has_mortgage": holds["mortgage"],
        "has_investment": holds["investment"]})

    # --- cards ---
    card_rows = []
    for ctype, mask in (("debit", holds["debit_card"]), ("credit", holds["credit_card"])):
        ids = np.repeat(cid[mask], rng.integers(1, 3, mask.sum()))
        limit = np.where(ctype == "credit",
                         np.round(rng.lognormal(12.3, 0.6, len(ids)), -3), 0.0)
        card_rows.append(pd.DataFrame({
            "client_id": ids, "card_type": ctype,
            "payment_system": rng.choice(["visa", "mastercard", "mir"], len(ids), p=[0.38, 0.33, 0.29]),
            "open_date": _dates(rng, len(ids)), "is_blocked": rng.random(len(ids)) < 0.025,
            "credit_limit": limit if ctype == "credit" else 0.0}))
    cards = pd.concat(card_rows, ignore_index=True)
    cards.insert(0, "card_id", np.arange(1, len(cards) + 1))

    # --- monthly marts: txn, cashback, app activity ---
    mrep = np.tile(MONTHS, N_CLIENTS)
    crep = np.repeat(cid, len(MONTHS))
    # per-CLIENT base check with a heavy tail (so ">50k avg check" audiences are
    # non-empty), per-month jitter on top
    client_check = rng.lognormal(7.3, 1.3, N_CLIENTS) * seg_mult ** 0.9
    active_m = rng.random(len(crep)) < 0.88
    cnt = np.where(active_m, rng.poisson(np.repeat(np.clip(client_check / 300, 3, 60), len(MONTHS))), 0)
    check_m = np.repeat(client_check, len(MONTHS)) * rng.uniform(0.7, 1.3, len(crep))
    amount = np.round(np.where(cnt > 0, cnt * check_m, 0), 2)
    txn = pd.DataFrame({
        "client_id": crep, "month": mrep, "txn_cnt": cnt, "txn_amount": amount,
        "avg_check": np.round(np.where(cnt > 0, check_m, 0), 2),
        "online_share": np.round(np.where(cnt > 0, rng.beta(5, 3, len(crep)), 0), 3)})

    never_user = rng.random(N_CLIENTS) < 0.52          # clients who never spend cashback
    earned = np.round(amount * 0.01 * (rng.random(len(crep)) < 0.9), 2)
    used = np.round(np.where(np.repeat(never_user, len(MONTHS)), 0,
                             earned * rng.beta(2, 2, len(crep)) * (rng.random(len(crep)) < 0.6)), 2)
    cashback = pd.DataFrame({"client_id": crep, "month": mrep,
                             "cashback_earned": earned, "cashback_used": used})

    heavy = rng.random(N_CLIENTS) < 0.33
    logins = rng.poisson(np.repeat(np.where(heavy, 28, 8), len(MONTHS)))
    app = pd.DataFrame({
        "client_id": crep, "month": mrep, "app_logins": logins,
        "push_enabled": np.repeat(rng.random(N_CLIENTS) < 0.72, len(MONTHS)),
        "last_active_date": [m + pd.Timedelta(days=int(d)) if l > 0 else None
                             for m, d, l in zip(pd.to_datetime(mrep), rng.integers(0, 28, len(crep)), logins)]})

    # --- deposits, credits ---
    dep_ids = np.repeat(cid[holds["deposit"]], rng.integers(1, 3, holds["deposit"].sum()))
    deposits = pd.DataFrame({
        "deposit_id": np.arange(1, len(dep_ids) + 1), "client_id": dep_ids,
        "amount": np.round(rng.lognormal(13.3, 1.0, len(dep_ids)), 2),
        "currency": rng.choice(["RUB", "USD", "EUR"], len(dep_ids), p=[0.90, 0.06, 0.04]),
        "open_date": _dates(rng, len(dep_ids)),
        "rate": np.round(rng.uniform(0.05, 0.18, len(dep_ids)), 3),
        "is_active": rng.random(len(dep_ids)) < 0.9})

    cr_rows = []
    for p in ("mortgage", "auto_loan", "cash_loan"):
        ids = cid[holds[p]]
        principal = np.round(rng.lognormal({"mortgage": 15.0, "auto_loan": 13.8, "cash_loan": 12.6}[p],
                                           0.5, len(ids)), -3)
        cr_rows.append(pd.DataFrame({
            "client_id": ids, "product": p, "principal": principal,
            "outstanding": np.round(principal * rng.uniform(0.05, 0.95, len(ids)), 2),
            "open_date": _dates(rng, len(ids)), "is_overdue": rng.random(len(ids)) < 0.045}))
    credits = pd.concat(cr_rows, ignore_index=True)
    credits.insert(0, "credit_id", np.arange(1, len(credits) + 1))

    consent = pd.DataFrame({
        "client_id": cid, "sms_ok": rng.random(N_CLIENTS) < 0.70,
        "email_ok": rng.random(N_CLIENTS) < 0.58, "push_ok": rng.random(N_CLIENTS) < 0.75,
        "call_ok": rng.random(N_CLIENTS) < 0.30, "consent_updated": _dates(rng, N_CLIENTS)})

    scores = pd.DataFrame({
        "client_id": cid, "monthly_income_est": income,
        "churn_score": np.round(rng.beta(2.2, 4.5, N_CLIENTS), 4),
        "ltv_est": np.round(income * rng.uniform(4, 30, N_CLIENTS), -2),
        "credit_score": np.clip(rng.normal(660, 90, N_CLIENTS).astype(int), 300, 850)})

    return {"dm_client_profile": profile, "dm_client_products": products, "dm_card": cards,
            "dm_txn_monthly": txn, "dm_cashback_monthly": cashback, "dm_deposit": deposits,
            "dm_credit": credits, "dm_app_activity_monthly": app,
            "dm_marketing_consent": consent, "dm_client_scores": scores}


def sync_golden(db_path: Path, golden_path: Path) -> None:
    import yaml
    conn = duckdb.connect(str(db_path), read_only=True)
    text = golden_path.read_text(encoding="utf-8")
    items = yaml.safe_load(text)["items"]
    for it in items:
        n = len({r[0] for r in conn.execute(
            f"SELECT client_id FROM ({it['reference_sql'].rstrip(';')})").fetchall()})
        # rewrite the expected_size line inside this item's block, keyed by id
        text = re.sub(rf"(- id: {it['id']}\n(?:.*\n)*?\s*expected_size:)\s*\d+",
                      rf"\g<1> {n}", text)
        print(f"  {it['id']}: expected_size -> {n}")
    golden_path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "warehouse.duckdb"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite an existing DB")
    ap.add_argument("--sync-golden", action="store_true",
                    help="rewrite expected_size in eval/golden_set.yaml from the new DB")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists; the shipped DB is canonical for the golden set.\n"
              f"Use --force to regenerate (then --sync-golden + python -m eval.selfcheck).")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    tables = generate(args.seed)
    conn = duckdb.connect(str(out))
    for name, df in tables.items():
        conn.register("df", df)
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        conn.unregister("df")
        print(f"  {name:28s} {len(df):>7d} rows")
    conn.close()
    print(f"warehouse -> {out}")

    if args.sync_golden:
        print("syncing golden set expected sizes:")
        sync_golden(out, ROOT / "eval" / "golden_set.yaml")
        print("now run: python -m eval.selfcheck")
    return 0


if __name__ == "__main__":
    sys.exit(main())
