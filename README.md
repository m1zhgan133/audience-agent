# Audience Agent — baseline

Text-to-SQL агент для подбора клиентских аудиторий. Аналитик описывает сегмент
на естественном языке — агент подбирает витрины, пишет SQL, выполняет его read-only
и возвращает аудиторию (client_id) + выгрузку. Подход: 10 курируемых витрин в контексте, один SQL-инструмент, цикл
с самокоррекцией по ошибке, self-hosted модель по OpenAI-совместимому API, оценка на
golden set тремя метриками.

Это **baseline** — сознательно простой: без RAG по схеме, без few-shot, без reranking.
Оркестрация — **LangGraph** (StateGraph: `agent` → `run_sql`/`nudge`, условные рёбра).

**Измерено** (qwen3:8b через Ollama, golden set = 22): exact match **86.4%**,
mean Jaccard **0.99**, size MAE **1.68**, crash rate **0%**. Все промахи —
неоднозначность формулировок (окно усреднения, лишний `is_active`), не ошибки SQL.
Отчёт: `eval/reports/latest.json`.

## Что внутри

```
audience_agent/
  config.py          конфиг из ENV (endpoint, модель, лимиты) — Ollama по умолчанию
  sql_guard.py       валидация: только один SELECT/WITH, запрет DDL/DML
  db.py              read-only DuckDB: run_preview / audience_set
  prompt.py          system-промпт из описаний витрин
  agent.py           LangGraph-граф: agent -> run_sql / nudge, цикл до FINAL ANSWER
  server.py          FastAPI: /api/query, /api/execute, /api/export + UI
  schema/marts.yaml  10 витрин: бизнес-смысл, гранулярность, enum-значения
  static/index.html  минимальный чат-интерфейс (показ SQL, правка, экспорт CSV)
scripts/
  generate_data.py   детерминированный синтетический банк -> data/warehouse.duckdb
  serve_vllm.sh      запуск Qwen3-30B-A3B через vLLM (нужен CUDA GPU)
  serve_ollama.sh    локальный путь на Mac (Metal, OpenAI-совместимый API)
  smoke_agent.py     e2e-тест цикла агента на фейковом LLM (без модели)
eval/
  golden_set.yaml    22 пары «запрос -> эталонный SQL (client_id)» + expected_size
  metrics.py         exact match, MAE/MSE по размеру, Jaccard/precision/recall по составу
  selfcheck.py       валидация данных+guard+метрик+golden set БЕЗ модели
  run_eval.py        прогон агента по golden set + агрегатные метрики (нужна модель)
```

## Быстрый старт (без модели — проверка каркаса)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/generate_data.py      # соберёт data/warehouse.duckdb (~5000 клиентов)
.venv/bin/python -m eval.selfcheck             # 22/22 golden queries + метрики
.venv/bin/python scripts/smoke_agent.py        # цикл агента на фейковом LLM
```

Всё это работает офлайн.

## Запуск модели

Агент provider-agnostic: любой OpenAI-совместимый endpoint (`OPENAI_BASE_URL`).

### Локально на Mac — Ollama
```bash
brew install ollama && ollama serve &
ollama pull qwen3:30b-a3b     # ~18GB, на 24GB RAM ТЕСНО; для итераций берите qwen3:8b
# config уже смотрит на Ollama по умолчанию, .env не нужен
```

### На GPU-сервере — vLLM
```bash
pip install "vllm>=0.6"
bash scripts/serve_vllm.sh                       # :8000, Qwen3-30B-A3B
export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=not-needed MODEL=qwen3-30b-a3b
```
> vLLM требует CUDA-GPU и не запустится на Mac (нет Metal-бэкенда). На Mac используйте Ollama.

## Живой прогон

```bash
# веб-интерфейс
.venv/bin/python -m uvicorn audience_agent.server:app --port 8080   # -> http://localhost:8080

# один запрос из CLI
.venv/bin/python -m audience_agent.agent "клиенты из Москвы с кредиткой и средним чеком за 3 месяца выше 50000"

# полный прогон по golden set с метриками
.venv/bin/python -m eval.run_eval               # отчёт -> eval/reports/latest.json
```

## Данные

Синтетический банк, детерминированно (seed=42), готовая база лежит в `data/warehouse.duckdb`, период Jan–Jun 2026. 10 витрин, ~5000
клиентов. Денормализованные флаги в `dm_client_profile` согласованы с детальными витринами,
поэтому разные формулировки одного сегмента дают одну аудиторию — на этом и держится
осмысленность оценки.

## Метрики

| Метрика | Что ловит |
|---|---|
| exact match | наборы клиентов совпали ровно |
| MAE/MSE по размеру | насколько промахнулись по численности аудитории |
| Jaccard / precision / recall по составу | *те ли* клиенты, а не только *столько ли* |

Jaccard/precision/recall ловят случай «размер совпал, состав нет».

## Безопасность

- read-only DuckDB-соединение (движок сам блокирует запись);
- `sql_guard`: единственный SELECT/WITH, запрет DDL/DML до попадания в БД;
- превью ограничено, экспорт — локальный CSV (для продакшена можно заменить на загрузку в объектное хранилище).
