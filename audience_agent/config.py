"""Central config. Everything overridable via env vars, so the same code runs
against Ollama on a laptop and vLLM on a Linux box (both OpenAI-compatible)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


@dataclass
class Config:
    # LLM endpoint (OpenAI-compatible). Ollama locally; vLLM in prod — same code.
    base_url: str = _env("OPENAI_BASE_URL", "LLM_BASE_URL", default="http://localhost:11434/v1")
    api_key: str = _env("OPENAI_API_KEY", "LLM_API_KEY", default="ollama")
    model: str = _env("MODEL", "LLM_MODEL", default="qwen3:30b-a3b")
    temperature: float = float(_env("LLM_TEMPERATURE", default="0.2"))
    # Qwen3 soft switch: appending /no_think disables thinking. Baseline keeps it
    # off — audience queries need schema fidelity, not long reasoning chains.
    think: bool = _env("LLM_THINK", default="0") == "1"

    max_iters: int = int(_env("AGENT_MAX_ITERS", default="8"))
    query_timeout_s: int = int(_env("SQL_TIMEOUT_S", default="30"))
    preview_rows: int = int(_env("PREVIEW_ROWS", default="20"))
    # hard cap on rows materialised for a preview; protects against a runaway cross join
    preview_hardcap: int = int(_env("PREVIEW_HARDCAP", default="5000"))

    db_path: str = _env("DB_PATH", default=str(ROOT / "data" / "warehouse.duckdb"))
    marts_path: str = _env("MARTS_PATH", default=str(ROOT / "audience_agent" / "schema" / "marts.yaml"))
    export_dir: str = _env("EXPORT_DIR", default=str(ROOT / "exports"))


CONFIG = Config()
