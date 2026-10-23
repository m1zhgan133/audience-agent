#!/usr/bin/env bash
# One-shot: set up venv, build the warehouse, run the model-free checks, start the UI.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

if [ ! -f data/warehouse.duckdb ]; then .venv/bin/python scripts/generate_data.py; fi

echo "== self-check (no model needed) =="; .venv/bin/python -m eval.selfcheck
echo "== agent smoke test (fake LLM) ==";  .venv/bin/python scripts/smoke_agent.py

echo
echo "Starting UI on http://localhost:8080  (needs a model endpoint for live queries)"
.venv/bin/python -m uvicorn audience_agent.server:app --port 8080
