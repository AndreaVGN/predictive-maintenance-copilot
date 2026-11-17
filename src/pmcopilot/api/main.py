"""
main.py — l'applicazione FastAPI: avvio, sicurezza, endpoint principali.

MAPPA DEGLI ENDPOINT
    Pubblici (senza login)
        GET  /health/live        il processo è vivo? (probe di liveness)
        GET  /health/ready       è pronto a servire? DB, modello, indice (readiness)
        GET  /                   console web (mostra il login se serve)
        POST /auth/login | /auth/logout,  GET /auth/me
    Utenti autenticati
        GET  /health             stato dettagliato dei componenti
        POST /predict            solo il modello ML
        POST /chat, /chat/stream domande all'agente
        POST /feedback           pollice su/giù su una risposta
        ...dati per la console (ui_routes.py)
    Solo admin
        POST /chat/{id}/decision[/stream]   approvare o rifiutare un'azione
        /experiments/*, /ops/*              valutazioni, drift, modelli (admin_routes.py)

LIVENESS vs READINESS (linguaggio di Kubernetes, ma il concetto è generale)
    liveness   "sei vivo?" Se fallisce, l'orchestratore RIAVVIA il container.
               Deve essere banale: se dipendesse da Ollama, un Ollama lento
               farebbe riavviare l'API in un ciclo senza fine.
    readiness  "puoi ricevere traffico?" Se fallisce, il bilanciatore smette di
               mandare richieste a questa istanza finché non torna pronta.

GESTIONE DEGLI ERRORI
    Un'eccezione imprevista diventa un 500 con un messaggio generico e il
    request_id: i dettagli finiscono nel log, non all'utente (lo stack trace
    può rivelare percorsi, versioni, query).
"""

from __future__ import annotations

import logging
import re
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pmcopilot.agent.service import ChatResult, CopilotService
from pmcopilot.api.admin_routes import router as admin_router
from pmcopilot.api.auth import (
    COOKIE_NAME,
    Authenticator,
    RateLimiter,
    User,
    check_limits,
    current_user,
    require_admin,
    verify_credentials,
)
from pmcopilot.api.schemas import (
    ChatRequest,
    ChatResponse,
    DecisionRequest,
    FeedbackRequest,
    HealthResponse,
    LoginRequest,
    PredictRequest,
    PredictResponse,
)
from pmcopilot.api.ui_routes import STATIC_DIR, result_payload
from pmcopilot.api.ui_routes import router as ui_router
from pmcopilot.config import Settings, get_settings
from pmcopilot.ml.predictor import FailurePredictor, MachineReading
from pmcopilot.observability.structured_logging import request_id_var, setup_logging_from_settings
from pmcopilot.observability.tracing import flush_langfuse, langfuse_available, langfuse_score
from pmcopilot.ops.jobs import build_scheduler
from pmcopilot.store import get_store

logger = logging.getLogger("pmcopilot.api")

CHART_NAME_PATTERN = r"^[A-Za-z0-9_\-]+\.png$"  # niente "../": blocca il path traversal


def ollama_reachable(settings: Settings) -> bool | None:
    """True/False per Ollama; None se il provider è un'API cloud (non la controlliamo)."""
    if settings.llm_provider != "ollama":
        return None
    try:
        with urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:
        return False


def enforce_password_policy(settings: Settings) -> None:
    if not settings.uses_default_password:
        return
    message = ("La password di admin è quella di default ('admin'). Impostala nel .env con "
               "PMC_ADMIN_PASSWORD prima di esporre il servizio.")
    if settings.env == "prod" and not settings.allow_default_password:
        raise RuntimeError(message + " In produzione l'avvio è bloccato.")
    logger.warning(message)


def create_app(settings: Settings | None = None, service: CopilotService | None = None,
               start_scheduler: bool | None = None) -> FastAPI:
    """Factory dell'app: i test le passano settings e un servizio con LLM finto."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging_from_settings(settings)
        enforce_password_policy(settings)
        app.state.settings = settings
        app.state.store = get_store(settings.app_db_path)
        app.state.auth = Authenticator(settings)
        app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
        app.state.predictor = FailurePredictor(settings.model_path)
        app.state.service = service or CopilotService.from_settings(settings)
        app.state.scheduler = build_scheduler(app)
        run_jobs = settings.scheduler_enabled if start_scheduler is None else start_scheduler
        if run_jobs:
            app.state.scheduler.start()
        logger.info("API pronta (env %s, modello ML %s, LLM %s, langfuse %s)", settings.env,
                    app.state.predictor.version, settings.llm_spec,
                    langfuse_available(settings) if settings.langfuse_enabled else "off")
        yield
        app.state.scheduler.stop()
        flush_langfuse(settings)
        logger.info("API in chiusura")

    app = FastAPI(title="Predictive Maintenance Copilot", version="0.2.0", lifespan=lifespan,
                  description="Agentic RAG + modello ML per la manutenzione predittiva. "
                              "Autenticazione: cookie di sessione (console) o HTTP Basic.")

    # ----------------------------------------------------------- middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
            request_id = uuid.uuid4().hex[:12]  # non fidarsi di un header arbitrario nei log
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("Errore non gestito su %s %s", request.method, request.url.path)
                response = JSONResponse({"detail": "Errore interno.", "request_id": request_id},
                                        status_code=500)
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            response.headers["X-Request-ID"] = request_id
            # Intestazioni di sicurezza di base per la console web.
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("Referrer-Policy", "same-origin")
            if not request.url.path.startswith(("/health", "/static")):
                # Ancora dentro il contesto della richiesta: il log porta il suo request_id.
                logger.info("%s %s -> %s", request.method, request.url.path, response.status_code,
                            extra={"fields": {"latency_ms": latency_ms, "status": response.status_code}})
            return response
        finally:
            request_id_var.reset(token)

    # ----------------------------------------------------------- pubblici
    @app.get("/health/live", tags=["health"])
    def live() -> dict:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request) -> JSONResponse:
        s: Settings = request.app.state.settings
        checks = {"database": s.db_path.exists(), "model": s.model_path.exists(),
                  "vector_index": s.chroma_dir.exists()}
        ok = all(checks.values())
        return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks},
                            status_code=200 if ok else 503)

    @app.post("/auth/login", tags=["auth"])
    def login(body: LoginRequest, request: Request, response: Response) -> dict:
        auth: Authenticator = request.app.state.auth
        client = request.client.host if request.client else "unknown"
        auth.check_login_allowed(client)
        user = verify_credentials(request.app.state.settings, body.username, body.password)
        if user is None:
            auth.record_failure(client)
            request.app.state.store.audit(body.username[:50], "login_failed", client)
            raise HTTPException(401, "Utente o password non corretti.")
        response.set_cookie(COOKIE_NAME, auth.make_cookie(user), httponly=True, samesite="lax",
                            secure=request.app.state.settings.env == "prod",
                            max_age=request.app.state.settings.session_hours * 3600)
        request.app.state.store.audit(user.name, "login", client)
        return {"user": user.name, "role": user.role,
                "default_password": user.is_admin and request.app.state.settings.uses_default_password}

    @app.post("/auth/logout", tags=["auth"])
    def logout(response: Response) -> dict:
        response.delete_cookie(COOKIE_NAME)
        return {"ok": True}

    @app.get("/auth/me", tags=["auth"])
    def me(request: Request, user: User = Depends(current_user)) -> dict:
        return {"user": user.name, "role": user.role,
                "default_password": user.is_admin and request.app.state.settings.uses_default_password}

    # ------------------------------------------------------- autenticati
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health(request: Request, user: User = Depends(current_user)) -> HealthResponse:
        s: Settings = request.app.state.settings
        checks = {"database": s.db_path.exists(), "model": s.model_path.exists(),
                  "vector_index": s.chroma_dir.exists()}
        ollama = ollama_reachable(s)
        if ollama is not None:
            checks["ollama"] = ollama
        if s.langfuse_enabled:
            checks["langfuse"] = langfuse_available(s)
        return HealthResponse(status="ok" if all(checks.values()) else "degraded", checks=checks,
                              model_version=request.app.state.predictor.version, llm_model=s.llm_spec)

    @app.post("/predict", response_model=PredictResponse, tags=["ml"])
    def predict(body: PredictRequest, request: Request, user: User = Depends(current_user)) -> PredictResponse:
        result = request.app.state.predictor.predict_one(MachineReading(**body.model_dump()))
        return PredictResponse(**result.to_dict())

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(body: ChatRequest, request: Request, user: User = Depends(current_user)) -> ChatResponse:
        check_limits(request, user)
        service: CopilotService = request.app.state.service
        if body.thread_id and service.has_pending(body.thread_id):
            raise HTTPException(409, "C'è un'approvazione in sospeso: va decisa prima di continuare.")
        try:
            result = service.chat(body.message, body.thread_id, user=user.name)
        except Exception as exc:
            logger.exception("Errore durante la chat")
            raise HTTPException(503, f"Il copilota non è disponibile: {type(exc).__name__}") from exc
        return _to_response(result)

    @app.post("/chat/{thread_id}/decision", response_model=ChatResponse, tags=["chat"])
    def decision(thread_id: str, body: DecisionRequest, request: Request,
                 user: User = Depends(require_admin)) -> ChatResponse:
        try:
            result = request.app.state.service.decide(thread_id, body.approve, body.message, user=user.name)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _to_response(result)

    @app.post("/feedback", tags=["chat"])
    def feedback(body: FeedbackRequest, request: Request, user: User = Depends(current_user)) -> dict:
        store = request.app.state.store
        store.add_feedback(user.name, body.request_id, body.thread_id, body.rating, body.comment)
        rows = store.query("SELECT trace_id FROM requests WHERE request_id = ? LIMIT 1", (body.request_id,))
        if rows and rows[0]["trace_id"]:
            langfuse_score(request.app.state.settings, rows[0]["trace_id"], "user_feedback",
                           float(body.rating), body.comment)
        return {"ok": True}

    @app.get("/charts/{filename}", tags=["chat"])
    def get_chart(filename: str, request: Request, user: User = Depends(current_user)) -> FileResponse:
        if not re.fullmatch(CHART_NAME_PATTERN, filename):
            raise HTTPException(400, "Nome file non valido.")
        path = request.app.state.settings.charts_dir / filename
        if not path.exists():
            raise HTTPException(404, "Grafico non trovato.")
        return FileResponse(path, media_type="image/png")

    app.include_router(ui_router)
    app.include_router(admin_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _to_response(result: ChatResult) -> ChatResponse:
    return ChatResponse(**result_payload(result))


# Oggetto usato da uvicorn: "uvicorn pmcopilot.api.main:app".
# create_app() non carica nulla finché il server non parte (lo fa il lifespan).
app = create_app()
