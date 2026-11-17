"""
tracing.py — vedere DENTRO l'agente: quali nodi, quali tool, quanto tempo, quanti token.

PERCHÉ SERVE
    Un agente è una scatola nera: la stessa domanda può generare 2 o 8 chiamate
    al modello, un tool può fallire e l'agente riprovare. Senza tracing, quando
    una risposta è sbagliata o lenta non sai perché.

DUE LIVELLI
    1. JsonlTraceHandler (sempre disponibile, zero dipendenze esterne)
       È un "callback handler" di LangChain: LangChain/LangGraph chiamano i
       suoi metodi on_*_start / on_*_end a ogni passo. Noi misuriamo i tempi
       e scriviamo una riga JSON per evento in logs/traces.jsonl.
       A fine richiesta `summary()` aggrega: latenza, chiamate LLM, token, tool.

    2. Langfuse (opzionale, consigliato)
       Piattaforma open source di LLM observability, installabile in locale con
       Docker. Mostra le trace come albero navigabile, con prompt, risposte,
       costi e la possibilità di dare punteggi. Si attiva con PMC_LANGFUSE_ENABLED=true
       e le chiavi LANGFUSE_* nel .env.

COME FUNZIONANO I CALLBACK
    Ogni "run" (nodo, tool, chiamata LLM) ha un run_id univoco. Allo start
    salviamo l'ora di inizio indicizzata per run_id; all'end calcoliamo la durata.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from pmcopilot.agent.guardrails import redact_pii
from pmcopilot.config import Settings

logger = logging.getLogger(__name__)
_FILE_LOCK = threading.Lock()  # un solo lock per il file: le richieste arrivano da thread diversi


def rotate_if_needed(path: Path, max_bytes: int, backups: int) -> None:
    """Rotazione dei log: traces.jsonl -> traces.jsonl.1 -> ... -> .N (il più vecchio si cancella).

    Senza rotazione un file di log cresce finché il disco si riempie: è una delle
    cause più banali (e più frequenti) di incidenti in produzione. Tenere N file
    di dimensione massima M limita lo spazio a N x M e definisce la retention.
    """
    if not path.exists() or path.stat().st_size < max_bytes:
        return
    for i in range(backups - 1, 0, -1):
        older = path.with_name(f"{path.name}.{i}")
        if older.exists():
            older.replace(path.with_name(f"{path.name}.{i + 1}"))
    path.replace(path.with_name(f"{path.name}.1"))
    oldest = path.with_name(f"{path.name}.{backups + 1}")
    if oldest.exists():
        oldest.unlink()


class JsonlTraceHandler(BaseCallbackHandler):
    """Registra nodi del grafo, tool e chiamate LLM su file JSONL."""

    def __init__(self, path: Path | None, request_id: str, thread_id: str,
                 redact: bool = True, max_bytes: int = 20 * 1024 * 1024, backups: int = 5):
        self.path = path
        self.redact = redact
        self.max_bytes = max_bytes
        self.backups = backups
        self.request_id = request_id
        self.thread_id = thread_id
        self._starts: dict[UUID, tuple[float, str, str]] = {}
        self._lock = threading.Lock()  # i callback possono arrivare da thread diversi
        self.started_at = time.perf_counter()
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        # Occupazione massima della finestra di contesto in questa richiesta:
        # prompt + risposta della chiamata al modello più "grande". Ollama riporta
        # in prompt_eval_count la dimensione TOTALE del prompt inviato (anche la
        # parte riusata dalla cache), quindi è la misura giusta del contesto usato.
        self.context_peak = 0
        self.tool_calls: Counter[str] = Counter()
        self.tool_errors = 0
        self.node_durations_ms: dict[str, float] = {}

    # ----------------------------------------------------------- scrittura
    def _write(self, event: str, name: str, **fields: Any) -> None:
        if self.path is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "event": event,
            "name": name,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        if self.redact:
            line = redact_pii(line)  # niente email/telefoni/codici fiscali nei file
        with _FILE_LOCK:
            rotate_if_needed(self.path, self.max_bytes, self.backups)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _start(self, run_id: UUID, kind: str, name: str) -> None:
        self._starts[run_id] = (time.perf_counter(), kind, name)

    def _end(self, run_id: UUID, **fields: Any) -> None:
        start = self._starts.pop(run_id, None)
        if start is None:
            return
        t0, kind, name = start
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        if kind == "node":
            self.node_durations_ms[name] = round(self.node_durations_ms.get(name, 0) + duration_ms, 1)
        self._write(f"{kind}_end", name, duration_ms=duration_ms, **fields)

    # ------------------------------------------------- nodi del grafo (chain)
    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: Any, *, run_id: UUID,
                       metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        # LangGraph mette il nome del nodo nei metadati: registriamo solo i nodi,
        # non ogni piccolo Runnable interno (sarebbe troppo rumore).
        node = (metadata or {}).get("langgraph_node")
        if node and kwargs.get("name") == node:
            self._start(run_id, "node", node)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        # GraphInterrupt (pausa per approvazione umana) passa da qui ma non è un errore.
        status = "interrupted" if type(error).__name__ == "GraphInterrupt" else "error"
        self._end(run_id, status=status, error=None if status == "interrupted" else repr(error))

    # --------------------------------------------------------------- tool
    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str, *, run_id: UUID,
                      **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self.tool_calls[name] += 1
        self._start(run_id, "tool", name)
        self._write("tool_start", name, input=input_str[:500])

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        text = str(getattr(output, "content", output))
        if text.startswith("ERRORE"):
            self.tool_errors += 1
        self._end(run_id, output_preview=text[:300])

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.tool_errors += 1
        self._end(run_id, status="error", error=repr(error))

    # ---------------------------------------------------------------- LLM
    def on_chat_model_start(self, serialized: dict[str, Any] | None, messages: list, *,
                            run_id: UUID, **kwargs: Any) -> None:
        self.llm_calls += 1
        self._start(run_id, "llm", (serialized or {}).get("name", "chat_model"))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.context_peak = max(self.context_peak, tokens_in + tokens_out)
        self._end(run_id, tokens_in=tokens_in, tokens_out=tokens_out)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, status="error", error=repr(error))

    # ------------------------------------------------------------ riepilogo
    def summary(self) -> dict[str, Any]:
        return {
            "latency_ms": round((time.perf_counter() - self.started_at) * 1000, 1),
            "llm_calls": self.llm_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "context_peak": self.context_peak,
            "tool_calls": dict(self.tool_calls),
            "tool_errors": self.tool_errors,
            "node_ms": self.node_durations_ms,
        }


# ------------------------------------------------------------------- Langfuse
# Controlliamo UNA volta se Langfuse risponde (e ricontrolliamo ogni minuto se
# era giù): se il server non è avviato l'app deve funzionare lo stesso, senza
# riempire i log di errori a ogni richiesta.
_LANGFUSE_RECHECK_S = 60.0
_langfuse_state: dict[str, Any] = {"ok": None, "checked_at": 0.0}
_langfuse_lock = threading.Lock()


def langfuse_host() -> str:
    return (os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
            or "http://localhost:3000").rstrip("/")


def _init_langfuse_client(settings: Settings) -> Any:
    """Crea (una volta) il client Langfuse con la maschera per i dati personali.

    `mask` viene applicata a TUTTO ciò che parte verso Langfuse (prompt,
    risposte, input e output dei tool): i dati personali non lasciano l'app.
    `release` = versione dei prompt: in Langfuse puoi filtrare per versione.
    """
    from langfuse import Langfuse

    from pmcopilot.agent.prompts import PROMPT_VERSION

    def mask(data: Any, **_: Any) -> Any:
        if isinstance(data, str):
            return redact_pii(data)
        if isinstance(data, dict):
            return {k: mask(v) for k, v in data.items()}
        if isinstance(data, list):
            return [mask(v) for v in data]
        return data

    return Langfuse(mask=mask if settings.redact_pii else None, environment=settings.env,
                    release=f"prompt-{PROMPT_VERSION}")


def langfuse_available(settings: Settings) -> bool:
    """True se Langfuse è abilitato, installato e raggiungibile con le chiavi giuste."""
    if not settings.langfuse_enabled:
        return False
    with _langfuse_lock:
        now = time.monotonic()
        if _langfuse_state["ok"] is not None and (
            _langfuse_state["ok"] or now - _langfuse_state["checked_at"] < _LANGFUSE_RECHECK_S
        ):
            return bool(_langfuse_state["ok"])
        try:
            client = _langfuse_state.get("client") or _init_langfuse_client(settings)
            _langfuse_state["client"] = client
            ok = bool(client.auth_check())
        except Exception as exc:  # pacchetto mancante, server spento, chiavi sbagliate
            logger.warning("Langfuse non raggiungibile su %s (%s): tracing solo su file.",
                           langfuse_host(), type(exc).__name__)
            ok = False
        _langfuse_state.update(ok=ok, checked_at=now)
        return ok


def langfuse_score(settings: Settings, trace_id: str | None, name: str, value: float,
                   comment: str | None = None) -> None:
    """Invia un punteggio a una trace (feedback utenti, voti del giudice). Mai bloccante."""
    if not trace_id or not langfuse_available(settings):
        return
    try:
        _langfuse_state["client"].create_score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception as exc:
        logger.warning("Score Langfuse non inviato: %s", exc)


def flush_langfuse(settings: Settings) -> None:
    """Invia subito le trace in coda (utile prima di chiudere un comando CLI)."""
    if _langfuse_state.get("ok") and settings.langfuse_enabled:
        try:
            _langfuse_state["client"].flush()
        except Exception:
            pass


def build_callbacks(settings: Settings, request_id: str, thread_id: str
                    ) -> tuple[JsonlTraceHandler, list, str | None, str | None]:
    """Crea i callback per una richiesta: il nostro handler + Langfuse se attivo.

    Restituisce anche link e id della trace su Langfuse (o None). Generiamo NOI
    l'id della trace: così conosciamo il link prima che la richiesta finisca e
    possiamo attaccarci dopo dei punteggi (feedback, voto del giudice).
    """
    tracer = JsonlTraceHandler(settings.trace_path if settings.trace_to_file else None,
                               request_id, thread_id, redact=settings.redact_pii,
                               max_bytes=settings.log_max_mb * 1024 * 1024, backups=settings.log_backups)
    callbacks: list = [tracer]
    trace_url = trace_id = None
    if langfuse_available(settings):
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            trace_id = Langfuse.create_trace_id()
            callbacks.append(CallbackHandler(trace_context={"trace_id": trace_id}))
            trace_url = f"{langfuse_host()}/project/{settings.langfuse_project_id}/traces/{trace_id}"
        except Exception as exc:
            logger.warning("Langfuse non disponibile (%s): continuo senza.", exc)
            trace_id = None
    return tracer, callbacks, trace_url, trace_id
