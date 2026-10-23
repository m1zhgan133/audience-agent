"""FastAPI backend + static UI for the audience agent.

Run:  uvicorn audience_agent.server:app --reload --port 8080
Then open http://localhost:8080
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from .agent import run_agent
from .config import CONFIG
from .sql_guard import SQLGuardError, guard

app = FastAPI(title="Audience Agent")
STATIC = Path(__file__).resolve().parent / "static"


class QueryIn(BaseModel):
    question: str


class SQLIn(BaseModel):
    sql: str


@app.get("/api/schema")
def schema():
    with open(CONFIG.marts_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    return [{"table": m["table"], "grain": m["grain"],
             "description": m["description"].strip()} for m in spec["marts"]]


@app.post("/api/query")
def query(inp: QueryIn):
    r = run_agent(inp.question)
    preview = None
    if r.audience_sql:
        try:
            preview = db.run_preview(r.audience_sql, preview_rows=20)
        except Exception:
            preview = None
    return {
        "answer": r.answer,
        "audience_sql": r.audience_sql,
        "audience_size": r.audience_size,
        "iterations": r.iterations,
        "preview": preview,
    }


@app.post("/api/execute")
def execute(inp: SQLIn):
    """Run an analyst-edited query read-only (the 'edit SQL and rerun' feature)."""
    try:
        clean = guard(inp.sql)
        res = db.run_preview(clean, preview_rows=50)
        return {"ok": True, **res}
    except SQLGuardError as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"guard: {e}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})


@app.post("/api/export")
def export(inp: SQLIn):
    """Baseline 'export to S3': write the full audience client_ids to a local CSV.

    Swap this body for an S3 put_object to match the production feature.
    """
    try:
        clean = guard(inp.sql)
        ids = sorted(db.audience_set(clean))
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    Path(CONFIG.export_dir).mkdir(parents=True, exist_ok=True)
    path = Path(CONFIG.export_dir) / "audience.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client_id"])
        for cid in ids:
            w.writerow([cid])
    return {"ok": True, "path": str(path), "count": len(ids)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
