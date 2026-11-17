"""
service.py — un'unica interfaccia per "parlare" con il copilota.

PERCHÉ UN LIVELLO DI SERVIZIO
    API, terminale (pmc chat), valutazione (pmc eval) fanno la stessa cosa:
    inviare un messaggio, gestire le approvazioni, ottenere risposta e
    metadati. La logica sta qui una volta sola; l'API è uno strato HTTP sopra.

COSA SUCCEDE A OGNI DOMANDA
    1. prepara callback (tracing JSONL + Langfuse) e config (thread, utente, tag)
    2. esegue il grafo (tutto insieme con invoke, o passo passo con stream)
    3. legge lo stato finale dal checkpointer: risposta, tool, fonti, interrupt
    4. calcola costo stimato e registra la richiesta in app.db (storico persistente)
    5. per le approvazioni: registra CHI ha deciso (audit) e firma l'ordine di lavoro

CHI È L'UTENTE
    Ogni chiamata riceve `user`: lo usa l'audit, il budget di token, Langfuse
    (filtrare le trace per utente) e lo storico delle richieste.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import closing
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from pmcopilot.agent.graph import build_graph
from pmcopilot.agent.llm import estimate_cost_usd, get_chat_model, get_embeddings
from pmcopilot.agent.prompts import PROMPT_VERSION
from pmcopilot.agent.tools import ToolContext
from pmcopilot.config import Settings
from pmcopilot.data.database import connect, list_machine_ids, migrate_plant_db
from pmcopilot.ml.predictor import FailurePredictor
from pmcopilot.observability.structured_logging import request_id_var
from pmcopilot.observability.tracing import build_callbacks
from pmcopilot.rag.retriever import ManualRetriever
from pmcopilot.store import get_store, now_iso

logger = logging.getLogger(__name__)


@dataclass
class PendingAction:
    tool: str
    args: dict[str, Any]
    description: str = ""


@dataclass
class ChatResult:
    thread_id: str
    answer: str
    route: str
    tools_used: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    guard_flags: list[str] = field(default_factory=list)
    pending_approval: list[PendingAction] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    tool_outputs: list[str] = field(default_factory=list)  # per LLM-as-a-judge
    request_id: str = ""
    langfuse_url: str | None = None
    langfuse_trace_id: str | None = None
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class CopilotService:
    def __init__(self, settings: Settings, ctx: ToolContext, llm: BaseChatModel,
                 router_llm: BaseChatModel | None = None, record: bool = True):
        self.settings = settings
        self.ctx = ctx
        self.record = record  # False per la valutazione: non sporca le metriche di esercizio
        self.store = get_store(settings.app_db_path)
        migrate_plant_db(settings.db_path)
        self.graph = build_graph(ctx, llm, router_llm=router_llm)

    @classmethod
    def from_settings(cls, settings: Settings, llm: BaseChatModel | None = None,
                      record: bool = True) -> CopilotService:
        ctx = ToolContext(
            settings=settings,
            predictor=FailurePredictor(settings.model_path),
            retriever=ManualRetriever(settings, get_embeddings(settings)),
            machine_ids=set(list_machine_ids(settings.db_path)),
        )
        return cls(settings, ctx, llm or get_chat_model(settings), record=record)

    def reload_model(self) -> str:
        """Ricarica il modello ML dal disco (dopo una promozione nel registry)."""
        self.ctx.predictor = FailurePredictor(self.settings.model_path)
        return self.ctx.predictor.version

    # ------------------------------------------------------------- internals
    def _prepare(self, thread_id: str, user: str, session_id: str | None,
                 tags: list[str] | None) -> tuple[str, Any, dict, str | None, str | None]:
        request_id = request_id_var.get()
        if request_id == "-":
            request_id = uuid.uuid4().hex[:12]
        tracer, callbacks, langfuse_url, trace_id = build_callbacks(self.settings, request_id, thread_id)
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks,
            "recursion_limit": 50,  # rete di sicurezza di LangGraph contro i cicli infiniti
            # Chiavi lette dall'integrazione Langfuse: raggruppano e filtrano le trace.
            "metadata": {"thread_id": thread_id, "app": "pmcopilot", "prompt_version": PROMPT_VERSION,
                         "llm": self.settings.llm_spec, "langfuse_user_id": user,
                         "langfuse_session_id": session_id or thread_id,
                         "langfuse_tags": [self.settings.env, self.settings.llm_provider, *(tags or [])]},
        }
        return request_id, tracer, config, langfuse_url, trace_id

    def _collect(self, thread_id: str, request_id: str, tracer: Any, config: dict,
                 langfuse_url: str | None, trace_id: str | None, user: str, kind: str,
                 error: str | None = None) -> ChatResult:
        snapshot = self.graph.get_state(config)
        values = snapshot.values

        pending: list[PendingAction] = []
        for intr in snapshot.interrupts:
            value = intr.value if isinstance(intr.value, dict) else {}
            for req in value.get("action_requests", []):
                pending.append(PendingAction(tool=req.get("name", "?"), args=req.get("args", {}),
                                             description=req.get("description", "")))
        if pending:
            answer = "Serve un'approvazione prima di procedere:\n" + "\n".join(
                f"- {p.tool}({', '.join(f'{k}={v!r}' for k, v in p.args.items())})" for p in pending)
        else:
            last = values["messages"][-1] if values.get("messages") else None
            answer = str(last.content) if isinstance(last, AIMessage) else ""

        tool_outputs: list[str] = []
        for m in reversed(values.get("messages", [])):
            if isinstance(m, HumanMessage):
                break
            if isinstance(m, ToolMessage):
                tool_outputs.insert(0, str(m.content))

        trace = tracer.summary()
        cost = estimate_cost_usd(self.settings, trace["tokens_in"], trace["tokens_out"])
        trace["cost_usd"] = cost
        result = ChatResult(
            thread_id=thread_id, answer=answer, route=values.get("route", "unknown"),
            tools_used=values.get("tools_used", []) if not pending else [],
            sources=values.get("sources", []), charts=values.get("charts", []),
            guard_flags=values.get("guard_flags", []), pending_approval=pending, trace=trace,
            tool_outputs=tool_outputs, request_id=request_id, langfuse_url=langfuse_url,
            langfuse_trace_id=trace_id, cost_usd=cost,
        )
        if self.record:
            self.store.record_request(
                request_id=request_id, user=user, thread_id=thread_id, kind=kind, route=result.route,
                latency_ms=trace["latency_ms"], llm_calls=trace["llm_calls"], tokens_in=trace["tokens_in"],
                tokens_out=trace["tokens_out"], context_peak=trace.get("context_peak", 0), cost_usd=cost,
                tools=trace["tool_calls"], guard_flags=result.guard_flags, error=error,
                llm=self.settings.llm_spec, prompt_version=PROMPT_VERSION, trace_id=trace_id)
        logger.info("chat completata", extra={"fields": {
            "thread_id": thread_id, "user": user, "route": result.route,
            "latency_ms": trace["latency_ms"], "llm_calls": trace["llm_calls"],
            "tools": trace["tool_calls"], "guard_flags": result.guard_flags, "pending": bool(pending)}})
        return result

    def _record_failure(self, request_id: str, user: str, thread_id: str, kind: str, exc: Exception,
                        started: float) -> None:
        if self.record:
            self.store.record_request(request_id=request_id, user=user, thread_id=thread_id, kind=kind,
                                      route="error", latency_ms=round((time.perf_counter() - started) * 1000, 1),
                                      llm_calls=0, tokens_in=0, tokens_out=0, context_peak=0, cost_usd=0.0,
                                      tools={}, guard_flags=[], error=f"{type(exc).__name__}: {exc}"[:500],
                                      llm=self.settings.llm_spec, prompt_version=PROMPT_VERSION)

    def _run(self, payload: Any, thread_id: str, user: str, kind: str,
             session_id: str | None = None, tags: list[str] | None = None) -> ChatResult:
        request_id, tracer, config, langfuse_url, trace_id = self._prepare(thread_id, user, session_id, tags)
        started = time.perf_counter()
        try:
            self.graph.invoke(payload, config)
        except Exception as exc:
            self._record_failure(request_id, user, thread_id, kind, exc, started)
            raise
        return self._collect(thread_id, request_id, tracer, config, langfuse_url, trace_id, user, kind)

    def _stream(self, payload: Any, thread_id: str, user: str, kind: str) -> Iterator[dict[str, Any]]:
        """Come _run, ma restituisce gli eventi man mano che il grafo avanza (per la UI).

        graph.stream(stream_mode="updates", subgraphs=True) produce, dopo ogni
        nodo, (namespace, {nodo: aggiornamento}). Il namespace è vuoto per il
        grafo esterno e contiene "agent:..." per i nodi del sottografo dell'agente.
        """
        request_id, tracer, config, langfuse_url, trace_id = self._prepare(thread_id, user, None, None)
        started = time.perf_counter()

        def event(kind_: str, **data: Any) -> dict[str, Any]:
            return {"type": kind_, "t_ms": round((time.perf_counter() - started) * 1000), **data}

        yield event("start", thread_id=thread_id, request_id=request_id, langfuse_url=langfuse_url)
        try:
            for namespace, update in self.graph.stream(payload, config, stream_mode="updates",
                                                       subgraphs=True):
                for node, delta in update.items():
                    if node.startswith("__") or "Middleware" in node or node == "system_prompt":
                        continue
                    yield from _events_for_node(node, delta or {}, bool(namespace), event)
        except Exception as exc:
            self._record_failure(request_id, user, thread_id, kind, exc, started)
            raise
        result = self._collect(thread_id, request_id, tracer, config, langfuse_url, trace_id, user, kind)
        if kind == "decision":
            self._after_decision(thread_id, user, result)
        yield event("final", result=result.to_dict())

    # ------------------------------------------------------------ approvazioni
    def _resume_command(self, thread_id: str, approve: bool, message: str | None) -> tuple[Command, list]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.interrupts:
            raise ValueError(f"Nessuna approvazione in sospeso per la conversazione {thread_id}.")
        actions = [req for i in snapshot.interrupts if isinstance(i.value, dict)
                   for req in i.value.get("action_requests", [])]
        decision: dict[str, Any] = ({"type": "approve"} if approve else
                                    {"type": "reject",
                                     "message": message or "L'utente ha rifiutato l'operazione. Non riproporla."})
        return Command(resume={"decisions": [decision] * max(len(actions), 1)}), actions

    def _after_decision(self, thread_id: str, user: str, result: ChatResult) -> None:
        """Firma gli ordini di lavoro creati con chi li ha approvati."""
        values = self.graph.get_state({"configurable": {"thread_id": thread_id}}).values
        for m in reversed(values.get("messages", [])):
            if isinstance(m, HumanMessage):
                break
            if isinstance(m, ToolMessage) and isinstance(m.artifact, dict) and m.artifact.get("work_order_id"):
                with closing(connect(self.settings.db_path)) as conn:
                    conn.execute("UPDATE work_orders SET approved_by = ?, approved_at = ? WHERE work_order_id = ?",
                                 (user, now_iso(), m.artifact["work_order_id"]))
                    conn.commit()

    def has_pending(self, thread_id: str) -> bool:
        return bool(self.graph.get_state({"configurable": {"thread_id": thread_id}}).interrupts)

    # --------------------------------------------------------------- API
    def chat(self, message: str, thread_id: str | None = None, *, user: str = "system",
             session_id: str | None = None, tags: list[str] | None = None) -> ChatResult:
        thread_id = thread_id or uuid.uuid4().hex
        return self._run({"messages": [HumanMessage(content=message)]}, thread_id, user, "chat",
                         session_id=session_id, tags=tags)

    def stream_chat(self, message: str, thread_id: str | None = None, *,
                    user: str = "system") -> Iterator[dict[str, Any]]:
        thread_id = thread_id or uuid.uuid4().hex
        return self._stream({"messages": [HumanMessage(content=message)]}, thread_id, user, "chat")

    def decide(self, thread_id: str, approve: bool, message: str | None = None, *,
               user: str = "system", session_id: str | None = None,
               tags: list[str] | None = None) -> ChatResult:
        command, actions = self._resume_command(thread_id, approve, message)
        self.store.audit(user, "approve" if approve else "reject", thread_id, actions=actions, message=message)
        result = self._run(command, thread_id, user, "decision", session_id=session_id, tags=tags)
        self._after_decision(thread_id, user, result)
        return result

    def stream_decide(self, thread_id: str, approve: bool, message: str | None = None, *,
                      user: str = "system") -> Iterator[dict[str, Any]]:
        command, actions = self._resume_command(thread_id, approve, message)  # errori subito
        self.store.audit(user, "approve" if approve else "reject", thread_id, actions=actions, message=message)
        return self._stream(command, thread_id, user, "decision")


# ---------------------------------------------------------------------------
# Traduzione degli aggiornamenti di LangGraph in eventi per la UI
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "input_guard": "Controllo della domanda",
    "injection_check": "Classificatore di injection",
    "router": "Verifica di pertinenza",
    "out_of_scope": "Domanda fuori ambito",
    "output_guard": "Controllo della risposta",
}
PREVIEW_CHARS = 1500


def _events_for_node(node: str, delta: dict[str, Any], in_agent: bool, event) -> Iterator[dict]:
    messages = delta.get("messages") or []
    if in_agent and node == "model":
        for msg in messages:
            if isinstance(msg, AIMessage):
                usage = msg.usage_metadata or {}
                yield event("model", tokens_in=usage.get("input_tokens", 0),
                            tokens_out=usage.get("output_tokens", 0),
                            decided="tool" if msg.tool_calls else "risposta")
                for call in msg.tool_calls:
                    yield event("tool_call", tool=call["name"], args=call["args"], call_id=call["id"])
                if not msg.tool_calls and msg.content:
                    yield event("model_answer", preview=str(msg.content)[:400])
        return
    if in_agent and node == "tools":
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = str(msg.content)
                artifact = msg.artifact if isinstance(msg.artifact, dict) else {}
                yield event("tool_result", tool=msg.name, call_id=msg.tool_call_id,
                            status="error" if content.startswith("ERRORE") else "ok",
                            preview=content[:PREVIEW_CHARS], truncated=len(content) > PREVIEW_CHARS,
                            guard_flags=artifact.get("guard_flags", []),
                            retrieval={k: artifact[k] for k in ("mode", "filter") if k in artifact})
        return
    if in_agent and node == "SummarizationMiddleware.before_model":
        return
    if in_agent or node == "agent":
        return
    if node == "injection_check" and not delta:
        return  # classificatore disattivato: nessun evento
    detail = {key: delta[key] for key in ("route", "guard_flags", "sources", "tools_used") if key in delta}
    yield event("node", node=node, label=NODE_LABELS.get(node, node), detail=detail)
