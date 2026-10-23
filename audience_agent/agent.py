"""Агент на LangGraph: цикл с одним SQL-инструментом.

Граф:
    agent --(есть ```sql)--> run_sql --> agent          (цикл самокоррекции)
    agent --(FINAL ANSWER / лимит итераций)--> END
    agent --(нет ни SQL, ни финала)--> nudge --> agent

Узел agent зовёт модель (любой OpenAI-совместимый endpoint через ChatOpenAI),
узел run_sql валидирует запрос (sql_guard) и выполняет его read-only; результат
или текст ошибки БД возвращается в контекст — это механизм самокоррекции.
Последний успешный запрос с колонкой client_id считается аудиторией.

Публичный API совместим со старым: run_agent(question, llm=None) -> AgentResult,
где llm — опциональный callable (messages -> str) для тестов без модели.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph

try:  # Python 3.9: TypedDict с генериками стабильнее из typing_extensions
    from typing_extensions import TypedDict
except ImportError:  # pragma: no cover
    from typing import TypedDict

from . import db
from .config import CONFIG
from .prompt import build_system_prompt
from .sql_guard import SQLGuardError, guard

# Callable, принимающий список сообщений и возвращающий текст ответа модели.
# По умолчанию — ChatOpenAI; тесты подставляют фейк.
LLMFn = Callable[[List[Dict[str, str]]], str]

_SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Qwen3 отвечает с <think>...</think>; срезаем, чтобы детект SQL/финала видел только ответ.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_sql(text: str) -> Optional[str]:
    m = _SQL_BLOCK.search(text)
    return m.group(1).strip() if m else None


def is_final(text: str) -> bool:
    return "final answer" in text.lower()


def default_llm() -> LLMFn:
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        base_url=CONFIG.base_url,
        api_key=CONFIG.api_key,
        model=CONFIG.model,
        temperature=CONFIG.temperature,
    )

    def call(messages: List[Dict[str, str]]) -> str:
        # ChatOpenAI принимает сообщения в OpenAI-формате (role/content) как есть
        return chat.invoke(messages).content or ""

    return call


def _format_result(res: Dict) -> str:
    cols = res["columns"]
    preview = res["rows"][:10]
    head = f"OK. {res['total']} rows. Columns: {cols}."
    if not preview:
        return head + " (no rows)"
    body = "\n".join(str(list(r)) for r in preview)
    return f"{head}\nPreview (first {len(preview)}):\n{body}"


class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    audience_sql: Optional[str]
    audience_size: Optional[int]
    iterations: int


def build_graph(llm: LLMFn):
    """Собрать и скомпилировать LangGraph-граф агента вокруг данного LLM."""

    def agent_node(state: AgentState) -> AgentState:
        text = _THINK_BLOCK.sub("", llm(state["messages"])).strip()
        state["messages"].append({"role": "assistant", "content": text})
        state["iterations"] += 1
        return state

    def run_sql_node(state: AgentState) -> AgentState:
        sql = extract_sql(state["messages"][-1]["content"])
        try:
            clean = guard(sql)
        except SQLGuardError as e:
            state["messages"].append(
                {"role": "user", "content": f"Query rejected: {e}. Fix and resend."})
            return state
        try:
            res = db.run_preview(clean)
        except Exception as e:  # ошибка DuckDB -> обратно в контекст для самокоррекции
            state["messages"].append(
                {"role": "user", "content": f"SQL error: {e}. Fix and resend."})
            return state
        if "client_id" in res["columns"]:
            state["audience_sql"] = clean
            state["audience_size"] = res["total"]
        state["messages"].append({"role": "user", "content": _format_result(res)})
        return state

    def nudge_node(state: AgentState) -> AgentState:
        state["messages"].append(
            {"role": "user",
             "content": "Reply with a ```sql block to run, or a line starting 'FINAL ANSWER:'."})
        return state

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]["content"]
        if extract_sql(last) is not None:
            return "run_sql"
        if is_final(last) or state["iterations"] >= CONFIG.max_iters:
            return END
        return "nudge"

    def route_continue(state: AgentState) -> str:
        return END if state["iterations"] >= CONFIG.max_iters else "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("run_sql", run_sql_node)
    g.add_node("nudge", nudge_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"run_sql": "run_sql", "nudge": "nudge", END: END})
    g.add_conditional_edges("run_sql", route_continue, {"agent": "agent", END: END})
    g.add_conditional_edges("nudge", route_continue, {"agent": "agent", END: END})
    return g.compile()


@dataclass
class AgentResult:
    question: str
    answer: str
    audience_sql: Optional[str]
    audience_size: Optional[int]
    iterations: int
    error: Optional[str] = None
    trace: List[Dict[str, str]] = field(default_factory=list)


def run_agent(question: str, llm: Optional[LLMFn] = None) -> AgentResult:
    llm = llm or default_llm()
    graph = build_graph(llm)
    # Qwen3 soft switch: /no_think экономит латентность на локальных прогонах.
    user_q = question if CONFIG.think else question + " /no_think"
    init: AgentState = {
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_q},
        ],
        "audience_sql": None,
        "audience_size": None,
        "iterations": 0,
    }
    # recursion_limit: на каждую итерацию агента приходится до 2 доп. узлов
    final = graph.invoke(init, config={"recursion_limit": CONFIG.max_iters * 3 + 4})

    last_assistant = ""
    for m in reversed(final["messages"]):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break

    return AgentResult(
        question=question,
        answer=last_assistant,
        audience_sql=final["audience_sql"],
        audience_size=final["audience_size"],
        iterations=final["iterations"],
        trace=final["messages"],
    )


def _cli() -> None:
    import sys
    q = " ".join(sys.argv[1:]) or "Клиенты из Москвы с кредитной картой"
    r = run_agent(q)
    print("Q:", r.question)
    print("Audience SQL:", r.audience_sql)
    print("Size:", r.audience_size, "| iterations:", r.iterations)
    print("Answer:", r.answer)


if __name__ == "__main__":
    _cli()
