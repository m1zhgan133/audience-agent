#!/usr/bin/env bash
# Serve Qwen3-30B-A3B via vLLM as an OpenAI-compatible endpoint on :8000.
#
# REQUIRES a CUDA GPU (Linux). This is the target the agent was designed for.
# Rough memory: bf16 ~60GB -> 1xA100-80G (or 2xA100-40G with tensor-parallel);
# FP8/AWQ fits smaller. It will NOT run on a Mac (no CUDA) — use serve_ollama.sh there.
#
#   pip install "vllm>=0.6"
#   bash scripts/serve_vllm.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
PORT="${PORT:-8000}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen3-30b-a3b \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --dtype auto
# Multi-GPU: add  --tensor-parallel-size 2
# Then point the agent at it:
#   export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=not-needed MODEL=qwen3-30b-a3b
