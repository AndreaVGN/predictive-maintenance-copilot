"""
admin_routes.py — endpoint per l'amministratore: esperimenti, monitoraggio, modelli.

ESPERIMENTI (/experiments)
    elenco dei run, dettaglio, confronto tra due run, andamento di una metrica
    nel tempo, quality gate, e lancio di nuove valutazioni dalla console.

OPERAZIONI (/ops)
    lavori periodici (stato ed "esegui ora"), drift, simulazione del flusso
    sensori, alert, registry dei modelli (promozione/rollback), audit, metriche.

LAVORI LUNGHI IN BACKGROUND
    Una valutazione dell'agente o un addestramento durano minuti: una richiesta
    HTTP non può restare appesa così a lungo (timeout di proxy e browser).
    L'endpoint avvia il lavoro in un thread e restituisce subito un id; la
    console interroga GET /ops/tasks/{id} finché lo stato non è "completato".
    Un solo lavoro pesante alla volta (max_workers=1): con Ollama locale,
    due valutazioni in parallelo si rallenterebbero a vicenda.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from pmcopilot.api.auth import User, current_user, require_admin
from pmcopilot.api.schemas import EvalRunRequest, SimulateRequest, TrainRequest
from pmcopilot.config import Settings
from pmcopilot.evaluation.compare import check_gate, compare_runs
from pmcopilot.ml.predictor import FailurePredictor
from pmcopilot.observability.metrics import compute_metrics, daily_series, prometheus_text
from pmcopilot.store import AppStore, now_iso

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pmc-task")
_tasks: dict[str, dict[str, Any]] = {}
_tasks_lock = threading.Lock()


def _submit(kind: str, user: str, func) -> dict[str, Any]:
    task_id = uuid.uuid4().hex[:10]
    task = {"id": task_id, "kind": kind, "user": user, "status": "in coda", "created": now_iso(),
            "result": None, "error": None}
    with _tasks_lock:
        _tasks[task_id] = task

    def run() -> None:
        task["status"] = "in corso"
        try:
            task["result"] = func()
            task["status"] = "completato"
        except Exception as exc:
            task["status"] = "errore"
            task["error"] = f"{type(exc).__name__}: {exc}"[:500]
        task["finished"] = now_iso()

    _executor.submit(run)
    return task


def _store(request: Request) -> AppStore:
    return request.app.state.store


# =============================================================== metriche
@router.get("/metrics", tags=["observability"])
def metrics(request: Request, minutes: int | None = Query(None, ge=1),
            user: User = Depends(current_user)) -> dict:
    return compute_metrics(_store(request), minutes)


@router.get("/metrics/daily", tags=["observability"])
def metrics_daily(request: Request, days: int = Query(14, ge=1, le=365),
                  user: User = Depends(require_admin)) -> list[dict]:
    return daily_series(_store(request), days)


@router.get("/metrics/prometheus", response_class=PlainTextResponse, tags=["observability"])
def metrics_prometheus(request: Request, user: User = Depends(current_user)) -> str:
    drift = _store(request).drift_reports(limit=1)
    extra = {}
    if drift:
        extra["pmc_drift_max_psi"] = max(f["psi"] for f in drift[0]["features"].values())
        extra["pmc_prediction_drift_psi"] = drift[0]["prediction"]["psi"]
    return prometheus_text(_store(request), extra)


# ============================================================ esperimenti
@router.get("/experiments/runs", tags=["experiments"])
def list_runs(request: Request, kind: str | None = None, user: User = Depends(require_admin)) -> list[dict]:
    return _store(request).list_eval_runs(kind)


@router.get("/experiments/runs/{run_id}", tags=["experiments"])
def get_run(run_id: str, request: Request, user: User = Depends(require_admin)) -> dict:
    run = _store(request).get_eval_run(run_id)
    if not run:
        raise HTTPException(404, "Run non trovato.")
    return run


@router.get("/experiments/compare", tags=["experiments"])
def compare(request: Request, a: str, b: str, user: User = Depends(require_admin)) -> dict:
    try:
        return compare_runs(_store(request), a, b)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/experiments/trend", tags=["experiments"])
def trend(request: Request, kind: str, user: User = Depends(require_admin)) -> list[dict]:
    """Tutti i run di un tipo, dal più vecchio: per il grafico dell'andamento."""
    runs = _store(request).list_eval_runs(kind, limit=500)
    return [{"id": r["id"], "ts": r["ts"], "label": r["label"], "summary": r["summary"],
             "prompt_version": r["prompt_version"], "llm": (r["config"] or {}).get("llm"),
             "retriever_mode": (r["config"] or {}).get("retriever_mode")}
            for r in reversed(runs)]


@router.get("/experiments/gate", tags=["experiments"])
def gate(request: Request, kind: str, run: str | None = None, baseline: str | None = None,
         user: User = Depends(require_admin)) -> dict:
    try:
        return check_gate(request.app.state.settings, _store(request), kind, run, baseline)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/experiments/run", tags=["experiments"])
def run_experiment(body: EvalRunRequest, request: Request, user: User = Depends(require_admin)) -> dict:
    settings: Settings = request.app.state.settings

    def job() -> Any:
        if body.kind == "guardrails":
            from pmcopilot.evaluation.guardrails import evaluate_guardrails

            return evaluate_guardrails(settings, with_llm=body.with_llm, label=body.label, user=user.name)
        if body.kind == "retrieval":
            from pmcopilot.evaluation.retrieval import MODES, evaluate_retrieval

            modes = MODES if body.mode == "all" else [body.mode or settings.retriever_mode]
            return [evaluate_retrieval(settings, mode=m, label=body.label and f"{body.label} ({m})",
                                       user=user.name) for m in modes]
        from pmcopilot.evaluation.agent import evaluate_agent

        return evaluate_agent(settings, limit=body.limit, label=body.label, user=user.name,
                              categories=body.categories)

    return _submit(f"eval:{body.kind}", user.name, job)


# ============================================================== operazioni
@router.get("/ops/tasks", tags=["ops"])
def tasks(user: User = Depends(require_admin)) -> list[dict]:
    with _tasks_lock:
        return sorted(_tasks.values(), key=lambda t: t["created"], reverse=True)[:50]


@router.get("/ops/tasks/{task_id}", tags=["ops"])
def task(task_id: str, user: User = Depends(require_admin)) -> dict:
    if task_id not in _tasks:
        raise HTTPException(404, "Lavoro non trovato.")
    return _tasks[task_id]


@router.get("/ops/jobs", tags=["ops"])
def jobs(request: Request, user: User = Depends(require_admin)) -> list[dict]:
    return request.app.state.scheduler.status()


@router.post("/ops/jobs/{name}/run", tags=["ops"])
def run_job(name: str, request: Request, user: User = Depends(require_admin)) -> dict:
    scheduler = request.app.state.scheduler
    if name not in scheduler.jobs:
        raise HTTPException(404, f"Lavoro '{name}' inesistente.")
    _store(request).audit(user.name, "run_job", name)
    job = scheduler.run_job(name)
    return {"name": name, "status": job.last_status, "result": job.last_result,
            "duration_ms": job.last_duration_ms}


@router.get("/ops/drift", tags=["ops"])
def drift_reports(request: Request, limit: int = Query(60, ge=1, le=500),
                  user: User = Depends(require_admin)) -> list[dict]:
    return _store(request).drift_reports(limit)


@router.post("/ops/simulate", tags=["ops"])
def simulate(body: SimulateRequest, request: Request, user: User = Depends(require_admin)) -> dict:
    """Aggiunge letture simulate, poi aggiorna scoring e controllo del drift."""
    from pmcopilot.ml.drift import check_drift
    from pmcopilot.ml.score import score_recent_readings
    from pmcopilot.ml.simulate import simulate_readings

    settings: Settings = request.app.state.settings
    summary = simulate_readings(settings, body.hours, body.scenario, body.intensity, body.ramp)
    _store(request).audit(user.name, "simulate", body.scenario, **body.model_dump())
    score_recent_readings(settings)
    report = check_drift(settings)
    return {"simulation": summary, "drift": {"status": report["status"],
                                             "diagnosis": report["prediction"]["diagnosis"]}}


@router.get("/ops/alerts", tags=["ops"])
def alerts(request: Request, user: User = Depends(require_admin)) -> list[dict]:
    return _store(request).alerts()


@router.get("/ops/audit", tags=["ops"])
def audit(request: Request, limit: int = Query(100, ge=1, le=1000),
          user: User = Depends(require_admin)) -> list[dict]:
    return _store(request).query("SELECT * FROM audit_log ORDER BY ts DESC, id DESC LIMIT ?", (limit,))


@router.get("/ops/models", tags=["ops"])
def models(request: Request, user: User = Depends(require_admin)) -> list[dict]:
    return _store(request).models()


@router.post("/ops/models/{version}/promote", tags=["ops"])
def promote_model(version: str, request: Request, user: User = Depends(require_admin)) -> dict:
    from pmcopilot.ml.registry import promote

    settings: Settings = request.app.state.settings
    try:
        card = promote(settings, version, user.name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    request.app.state.predictor = FailurePredictor(settings.model_path)
    request.app.state.service.reload_model()
    return {"promoted": version, "card": card}


@router.post("/ops/train", tags=["ops"])
def train(body: TrainRequest, request: Request, user: User = Depends(require_admin)) -> dict:
    settings: Settings = request.app.state.settings

    def job() -> dict:
        from pmcopilot.ml.train import train_and_evaluate

        result = train_and_evaluate(settings, quick=body.quick, split=body.split,
                                    promote_policy=body.promote_policy, user=user.name)
        if result.promoted:
            request.app.state.predictor = FailurePredictor(settings.model_path)
            request.app.state.service.reload_model()
        return {"version": result.model_version, "promoted": result.promoted,
                "pr_auc": result.test_metrics["pr_auc"],
                "champion_pr_auc": result.champion_metrics["pr_auc"] if result.champion_metrics else None}

    return _submit("train", user.name, job)
