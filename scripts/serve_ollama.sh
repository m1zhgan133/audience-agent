#!/usr/bin/env bash
# Local Mac (Apple Silicon) path — the realistic way to test on this machine.
# Ollama uses Metal and exposes an OpenAI-compatible API on :11434.
#
# qwen3:30b IS the 30B-A3B MoE (3B active). On a 24GB Mac it is TIGHT (~18GB, 4-bit):
# close other apps, keep the context small. For comfortable iteration use qwen3:8b.
set -euo pipefail

MODEL="${MODEL:-qwen3:30b}"   # or: qwen3:8b  (lighter),  qwen3:4b (fastest)

command -v ollama >/dev/null || { echo "Install Ollama first: brew install ollama"; exit 1; }
pgrep -x ollama >/dev/null || { echo "Start the server first: ollama serve"; exit 1; }

echo "Pulling $MODEL (this downloads several GB the first time)..."
ollama pull "$MODEL"

echo
echo "Point the agent at Ollama:"
echo "  export OPENAI_BASE_URL=http://localhost:11434/v1"
echo "  export OPENAI_API_KEY=ollama"
echo "  export MODEL=$MODEL"
