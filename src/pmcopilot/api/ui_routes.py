"""
ui_routes.py — endpoint per la console web (src/pmcopilot/api/static/), per utenti autenticati.

SERVER-SENT EVENTS (SSE)
    /chat/stream non restituisce un unico JSON alla fine: tiene aperta la
    risposta HTTP e invia una riga "data: {...}" per ogni evento, appena
    succede (nodo completato, tool chiamato, risultato del tool...).
    Il browser la legge con fetch() + ReadableStream e aggiorna la timeline.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from pmcopilot.agent.prompts import PROMPT_VERSION
from pmcopilot.agent.service import ChatResult, CopilotService
from pmcopilot.api.auth import User, check_limits, current_user, require_admin
from pmcopilot.api.schemas import ChatRequest, DecisionRequest
from pmcopilot.config import Settings
from pmcopilot.data.database import connect_readonly
from pmcopilot.observability.tracing import langfuse_available, langfuse_host

logger = logging.getLogger("pmcopilot.api.ui")

STATIC_DIR = Path(__file__).parent / "static"
FIGURE_NAME_PATTERN = r"^[A-Za-z0-9_\-]+\.png$"

router = APIRouter()


def result_payload(result: ChatResult) -> dict[str, Any]:
    """ChatResult -> dict per la UI/API (senza gli output grezzi dei tool, con URL dei grafici)."""
    data = result.to_dict()
    data.pop("tool_outputs", None)
    data.pop("langfuse_trace_id", None)
    data["charts"] = [f"/charts/{name}" for name in result.charts]
    return data


def query_records(settings: Settings, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Query in sola lettura su plant.db -> lista di dict pronta per il JSON (NaN -> null)."""
    with closing(connect_readonly(settings.db_path)) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return json.loads(df.to_json(orient="records"))


def _sse(events: Iterator[dict[str, Any]], request: Request) -> Iterator[str]:
    try:
        for ev in events:
            if ev["type"] == "final":
                result = ev["result"]
                result.pop("tool_outputs", None)
                result.pop("langfuse_trace_id", None)
                result["charts"] = [f"/charts/{name}" for name in result["charts"]]
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
    except Exception as exc:  # errore a metà stream: lo comunichiamo come evento
        logger.exception("Errore durante lo streaming")
        payload = {"type": "error", "message": f"{type(exc).__name__}: {exc}"[:300]}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _streaming_response(gen: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------- UI
@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/config", tags=["ui"])
def config(request: Request, user: User = Depends(current_user)) -> dict[str, Any]:
    s: Settings = request.app.state.settings
    source = s.data_source_path.read_text().strip() if s.data_source_path.exists() else "sconosciuta"
    return {
        "env": s.env,
        "llm_model": s.llm_spec,
        "embedding_model": f"{s.embedding_provider}:{s.embedding_model}",
        "retriever_mode": s.retriever_mode,
        "prompt_version": PROMPT_VERSION,
        "context_window": s.llm_num_ctx,
        "max_tokens_per_request": s.max_tokens_per_request,
        "daily_token_budget": s.daily_token_budget,
        "model_version": request.app.state.predictor.version,
        "model_threshold": request.app.state.predictor.threshold,
        "data_source": source,
        "router_enabled": s.router_enabled,
        "injection_classifier": s.injection_classifier,
        "langfuse": {"enabled": s.langfuse_enabled, "available": langfuse_available(s),
                     "url": f"{langfuse_host()}/project/{s.langfuse_project_id}"},
        "user": {"name": user.name, "role": user.role},
        "tokens_used_today": request.app.state.store.tokens_used_today(user.name),
    }


# ---------------------------------------------------------------- impianto
@router.get("/machines", tags=["ui"])
def machines(request: Request, user: User = Depends(current_user)) -> list[dict[str, Any]]:
    return query_records(request.app.state.settings, """
        SELECT m.machine_id, m.machine_type, m.line,
               f.n_readings, f.n_failures, f.twf, f.hdf, f.pwf, f.osf, f.rnf,
               r.failure_proba, r.risk_level,
               l.ts AS last_ts, l.air_temp_k, l.process_temp_k, l.rpm, l.torque_nm, l.tool_wear_min,
               (SELECT COUNT(*) FROM work_orders w
                 WHERE w.machine_id = m.machine_id AND w.status = 'aperto') AS open_work_orders
        FROM machines m
        LEFT JOIN v_failures_by_machine f ON f.machine_id = m.machine_id
        LEFT JOIN v_machine_risk r ON r.machine_id = m.machine_id
        LEFT JOIN v_latest_reading l ON l.machine_id = m.machine_id
        ORDER BY m.machine_id
    """)


@router.get("/machines/{machine_id}", tags=["ui"])
def machine_detail(machine_id: str, request: Request, hours: int = Query(48, ge=1, le=504),
                   user: User = Depends(current_user)) -> dict[str, Any]:
    s: Settings = request.app.state.settings
    machine_id = machine_id.upper()
    info = query_records(s, "SELECT * FROM machines WHERE machine_id = ?", (machine_id,))
    if not info:
        raise HTTPException(404, f"Macchina {machine_id} inesistente.")
    readings = query_records(s, """
        SELECT r.reading_id, r.ts, r.air_temp_k, r.process_temp_k, r.rpm, r.torque_nm,
               r.tool_wear_min, r.machine_failure, p.failure_proba
        FROM sensor_readings r LEFT JOIN predictions p ON p.reading_id = r.reading_id
        WHERE r.machine_id = ?
          AND r.ts >= datetime((SELECT MAX(ts) FROM sensor_readings WHERE machine_id = ?), ?)
        ORDER BY r.ts
    """, (machine_id, machine_id, f"-{hours} hours"))
    maintenance = query_records(s, """
        SELECT ts, failure_type, action, downtime_h, technician FROM maintenance_log
        WHERE machine_id = ? ORDER BY ts DESC LIMIT 10
    """, (machine_id,))
    orders = query_records(s, """
        SELECT work_order_id, created_at, failure_type, priority, notes, status, approved_by, approved_at
        FROM work_orders WHERE machine_id = ? ORDER BY created_at DESC LIMIT 10
    """, (machine_id,))
    return {"machine": info[0], "readings": readings, "maintenance": maintenance, "work_orders": orders}


@router.get("/model", tags=["ui"])
def model_card(request: Request, user: User = Depends(current_user)) -> dict[str, Any]:
    s: Settings = request.app.state.settings
    card = json.loads(s.model_card_path.read_text()) if s.model_card_path.exists() else {}
    figures = sorted(p.name for p in s.figures_dir.glob("*.png")) if s.figures_dir.exists() else []
    return {"card": card, "figures": figures}


@router.get("/reports/figures/{name}", include_in_schema=False)
def figure(name: str, request: Request, user: User = Depends(current_user)) -> FileResponse:
    if not re.fullmatch(FIGURE_NAME_PATTERN, name):
        raise HTTPException(400, "Nome file non valido.")
    path = request.app.state.settings.figures_dir / name
    if not path.exists():
        raise HTTPException(404, "Figura non trovata.")
    return FileResponse(path, media_type="image/png")


# --------------------------------------------------------------- streaming
@router.post("/chat/stream", tags=["chat"])
def chat_stream(body: ChatRequest, request: Request, user: User = Depends(current_user)) -> StreamingResponse:
    check_limits(request, user)
    service: CopilotService = request.app.state.service
    if body.thread_id and service.has_pending(body.thread_id):
        raise HTTPException(409, "C'è un'approvazione in sospeso: rispondi prima a quella.")
    return _streaming_response(_sse(service.stream_chat(body.message, body.thread_id, user=user.name), request))


@router.post("/chat/{thread_id}/decision/stream", tags=["chat"])
def decision_stream(thread_id: str, body: DecisionRequest, request: Request,
                    user: User = Depends(require_admin)) -> StreamingResponse:
    service: CopilotService = request.app.state.service
    try:
        events = service.stream_decide(thread_id, body.approve, body.message, user=user.name)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _streaming_response(_sse(events, request))
