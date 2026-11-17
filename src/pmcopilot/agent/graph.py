"""
graph.py — il cuore del progetto: il grafo LangGraph che orchestra tutto.

IL FLUSSO (diagramma nella guida, capitolo "L'agente")

    START
      │
      ▼
    input_guard ──(bloccato)───────────────────────────────► END
      │ ok                                    (risposta di rifiuto)
      ▼
    injection_check ──(attacco)────────────────────────────► END
      │ ok          (solo se PMC_INJECTION_CLASSIFIER=llm)
      ▼
    router ──(fuori ambito)──► out_of_scope ───────────────► END
      │ in ambito
      ▼
    agent  ◄── sottografo creato con create_agent (LangChain):
      │        ciclo "modello -> tool -> modello" con 5 tool e una pila di
      │        middleware (vedi sotto)
      ▼
    output_guard ──────────────────────────────────────────► END

LA PILA DI MIDDLEWARE DELL'AGENTE (dall'esterno verso l'interno)
    dynamic_prompt            system prompt ricalcolato a ogni chiamata
                              ("adesso" e macchine cambiano con i nuovi dati)
    SummarizationMiddleware   riassume la storia oltre il 70% del contesto
    HumanInTheLoopMiddleware  open_work_order si ferma finché un admin non decide
    ModelCallLimitMiddleware  massimo N chiamate al modello per domanda
    ToolCallLimitMiddleware   massimo N chiamate ai tool per domanda
    TokenBudgetMiddleware     massimo N token per domanda (nostro, middleware.py)
    ModelRetryMiddleware      ritenta le chiamate fallite con backoff esponenziale
    ModelFallbackMiddleware   se il modello principale è giù, prova quello di riserva
    ToolRetryMiddleware       ritenta i tool che sollevano eccezioni (es. DB occupato)
    ToolOutputGuardMiddleware ripulisce l'output dei tool (injection indiretta)

PERCHÉ UN GRAFO "ESTERNO" E UN AGENTE "INTERNO"
    create_agent è perfetto per il ciclo ReAct: decide da solo quali tool
    chiamare. Ma i controlli di sicurezza devono girare SEMPRE, prima e dopo:
    li mettiamo come nodi deterministici di uno StateGraph che avvolge l'agente.
    Controllo dove serve, autonomia dove aiuta.

MEMORIA E PAUSE
    Il checkpointer salva lo stato dopo ogni nodo, per thread_id. Con
    SqliteSaver lo stato è su disco: conversazioni e approvazioni in sospeso
    sopravvivono a un riavvio dell'API.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    dynamic_prompt,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from pmcopilot.agent import prompts
from pmcopilot.agent.guardrails import check_input, check_output
from pmcopilot.agent.llm import get_chat_model
from pmcopilot.agent.middleware import TokenBudgetMiddleware, ToolOutputGuardMiddleware
from pmcopilot.agent.tools import ToolContext, build_tools
from pmcopilot.config import Settings
from pmcopilot.data.database import dataset_time_range, describe_schema, list_machine_ids

logger = logging.getLogger(__name__)


# ==================================================================== stato
class CopilotState(MessagesState):
    """Stato del grafo: i messaggi (ereditati) + informazioni sul turno corrente."""

    route: str  # "pending" | "blocked" | "in_scope" | "out_of_scope"
    guard_flags: list[str]
    tools_used: list[str]
    sources: list[str]
    charts: list[str]


class RouteDecision(BaseModel):
    """Output strutturato del router: l'LLM deve restituire esattamente questo JSON."""

    route: Literal["in_scope", "out_of_scope"] = Field(description="Classificazione del messaggio")
    reason: str = Field(description="Motivazione in poche parole")


class InjectionVerdict(BaseModel):
    """Output strutturato del classificatore di prompt injection."""

    is_attack: bool = Field(description="True se il messaggio cerca di manipolare l'assistente")
    reason: str = Field(description="Motivazione in poche parole")


# ================================================================ helper
def _current_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """I messaggi dall'ultima domanda dell'utente in poi (il turno corrente)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def _last_user_text(messages: list[AnyMessage]) -> str:
    turn = _current_turn(messages)
    first = turn[0] if turn else None
    return str(first.content) if isinstance(first, HumanMessage) else ""


def build_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    """Memoria delle conversazioni: su disco (SQLite) o in RAM (test)."""
    if settings.checkpointer == "memory":
        return InMemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI esegue le richieste in thread diversi;
    # SqliteSaver protegge l'accesso con un suo lock interno.
    conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    return SqliteSaver(conn)


class _PromptCache:
    """Ricalcola il system prompt al massimo una volta al minuto (i dati cambiano poco)."""

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.text = ""
        self.at = 0.0

    def get(self) -> str:
        if time.monotonic() - self.at > 60 or not self.text:
            db = self.ctx.settings.db_path
            ids = list_machine_ids(db)
            self.ctx.machine_ids = set(ids)  # nuove macchine riconosciute anche dai guardrail
            first_ts, last_ts = dataset_time_range(db)
            self.text = prompts.build_agent_system_prompt(ids, first_ts, last_ts, describe_schema(db))
            self.at = time.monotonic()
        return self.text


# ================================================================ builder
def build_agent_middleware(settings: Settings, llm: BaseChatModel, prompt_cache: _PromptCache) -> list:
    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        return prompt_cache.get()

    middleware: list = [
        system_prompt,
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", int(settings.llm_num_ctx * settings.summarize_at_fraction)),
            keep=("messages", 12),
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={"open_work_order": {"allowed_decisions": ["approve", "reject"]}},
            description_prefix="Approvazione richiesta per aprire un ordine di lavoro",
        ),
        ModelCallLimitMiddleware(run_limit=settings.max_model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=settings.max_tool_calls, exit_behavior="continue"),
        TokenBudgetMiddleware(settings.max_tokens_per_request),
        ModelRetryMiddleware(max_retries=settings.llm_max_retries, initial_delay=1.0,
                             backoff_factor=2.0, max_delay=20.0),
    ]
    if settings.llm_fallback:
        middleware.append(ModelFallbackMiddleware(get_chat_model(settings, settings.llm_fallback)))
    middleware += [ToolRetryMiddleware(max_retries=1, initial_delay=0.5), ToolOutputGuardMiddleware()]
    return middleware


def build_graph(
    ctx: ToolContext,
    llm: BaseChatModel,
    router_llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    settings = ctx.settings
    prompt_cache = _PromptCache(ctx)

    agent = create_agent(
        model=llm,
        tools=build_tools(ctx),
        middleware=build_agent_middleware(settings, llm, prompt_cache),
        name="maintenance_agent",
    )
    small_llm = router_llm or llm
    structured_router = small_llm.with_structured_output(RouteDecision)
    injection_classifier = small_llm.with_structured_output(InjectionVerdict)

    # ------------------------------------------------------------ nodi
    def input_guard(state: CopilotState) -> dict:
        """Controllo deterministico della domanda (nessun LLM, istantaneo)."""
        result = check_input(_last_user_text(state["messages"]), settings.max_input_chars)
        reset = {"guard_flags": result.flags, "tools_used": [], "sources": [], "charts": []}
        if not result.allowed:
            logger.warning("Input bloccato: %s", result.flags)
            return {**reset, "route": "blocked", "messages": [AIMessage(content=result.reason or "")]}
        return {**reset, "route": "pending"}

    def injection_check(state: CopilotState) -> dict:
        """Secondo parere di un LLM sulle injection che le regex non vedono (parafrasi)."""
        if settings.injection_classifier != "llm":
            return {}
        text = _last_user_text(state["messages"])
        try:
            verdict = injection_classifier.invoke(prompts.INJECTION_CLASSIFIER_PROMPT.format(message=text))
        except Exception as exc:  # il classificatore è un aiuto: se fallisce non blocchiamo
            logger.warning("Classificatore injection non disponibile: %s", exc)
            return {"guard_flags": [*state["guard_flags"], "injection_classifier_error"]}
        if isinstance(verdict, InjectionVerdict) and verdict.is_attack:
            return {"route": "blocked", "guard_flags": [*state["guard_flags"], "injection:llm_classifier"],
                    "messages": [AIMessage(content="La richiesta sembra un tentativo di modificare "
                                                   "il comportamento dell'assistente.")]}
        return {}

    def router(state: CopilotState) -> dict:
        """L'LLM decide se la domanda è pertinente (output strutturato)."""
        if not settings.router_enabled:
            return {"route": "in_scope"}
        recent = state["messages"][-6:]
        history = "\n".join(f"{m.type}: {str(m.content)[:300]}" for m in recent
                            if isinstance(m, HumanMessage | AIMessage) and m.content)
        try:
            decision = structured_router.invoke(prompts.ROUTER_PROMPT.format(history=history))
            route = decision.route if isinstance(decision, RouteDecision) else "in_scope"
        except Exception as exc:  # meglio rispondere che bloccare per un errore tecnico
            logger.warning("Router non disponibile (%s): assumo in_scope", exc)
            return {"route": "in_scope", "guard_flags": [*state["guard_flags"], "router_error"]}
        return {"route": route}

    def out_of_scope(state: CopilotState) -> dict:
        return {"messages": [AIMessage(content=prompts.OUT_OF_SCOPE_REPLY)],
                "guard_flags": [*state["guard_flags"], "out_of_scope"]}

    def output_guard(state: CopilotState) -> dict:
        """Raccoglie tool/fonti/grafici/flag del turno e controlla la risposta finale."""
        turn = _current_turn(state["messages"])
        tool_msgs = [m for m in turn if isinstance(m, ToolMessage)]
        sources, charts, tool_flags = [], [], []
        for m in tool_msgs:
            artifact = m.artifact if isinstance(m.artifact, dict) else {}
            sources.extend(artifact.get("sources", []))
            tool_flags.extend(artifact.get("guard_flags", []))
            if artifact.get("chart"):
                charts.append(artifact["chart"])
        update: dict = {"tools_used": [m.name or "tool" for m in tool_msgs],
                        "sources": list(dict.fromkeys(sources)), "charts": charts}
        flags = [*state["guard_flags"], *dict.fromkeys(tool_flags)]
        final = state["messages"][-1]
        if isinstance(final, AIMessage):
            checked = check_output(str(final.content), known_machine_ids=ctx.machine_ids, sources=sources)
            flags += checked.flags
            if checked.answer != final.content:
                # Stesso id = add_messages SOSTITUISCE il messaggio invece di accodarlo.
                update["messages"] = [AIMessage(content=checked.answer, id=final.id)]
        update["guard_flags"] = flags
        return update

    # ------------------------------------------------- archi condizionali
    def after_input_guard(state: CopilotState) -> str:
        return END if state["route"] == "blocked" else "injection_check"

    def after_injection_check(state: CopilotState) -> str:
        return END if state["route"] == "blocked" else "router"

    def after_router(state: CopilotState) -> str:
        return "agent" if state["route"] == "in_scope" else "out_of_scope"

    builder = StateGraph(CopilotState)
    builder.add_node("input_guard", input_guard)
    builder.add_node("injection_check", injection_check)
    builder.add_node("router", router)
    builder.add_node("out_of_scope", out_of_scope)
    builder.add_node("agent", agent)
    builder.add_node("output_guard", output_guard)
    builder.add_edge(START, "input_guard")
    builder.add_conditional_edges("input_guard", after_input_guard, ["injection_check", END])
    builder.add_conditional_edges("injection_check", after_injection_check, ["router", END])
    builder.add_conditional_edges("router", after_router, ["agent", "out_of_scope"])
    builder.add_edge("out_of_scope", END)
    builder.add_edge("agent", "output_guard")
    builder.add_edge("output_guard", END)
    return builder.compile(checkpointer=checkpointer or build_checkpointer(settings))
