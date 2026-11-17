"""
jobs.py — collega i lavori periodici (scheduler.py) ai componenti dell'API.

Ogni lavoro è una funzione senza argomenti che usa lo stato dell'app
(settings, servizio, store). Le stesse operazioni esistono anche come
comandi `pmc` (score, drift check, backup, ingest), per chi preferisce
cron o un orchestratore esterno.
"""

from __future__ import annotations

from typing import Any

from pmcopilot.ops.scheduler import Scheduler


def build_scheduler(app: Any) -> Scheduler:
    settings = app.state.settings
    scheduler = Scheduler()

    def scoring() -> dict:
        from pmcopilot.ml.score import score_recent_readings

        return {"predictions": score_recent_readings(settings)}

    def drift() -> dict:
        from pmcopilot.ml.drift import check_drift

        report = check_drift(settings)
        return {"status": report["status"], "diagnosis": report["prediction"]["diagnosis"]}

    def alerts() -> dict:
        from pmcopilot.api.main import ollama_reachable
        from pmcopilot.observability.alerts import evaluate_rules

        results = evaluate_rules(settings, app.state.store, ollama_reachable(settings))
        return {"firing": [r["rule"] for r in results if r["firing"]]}

    def reindex() -> dict:
        from pmcopilot.agent.llm import get_embeddings
        from pmcopilot.rag.ingest import build_vector_store, manuals_changed

        if not manuals_changed(settings):
            return {"changed": False}
        stats = build_vector_store(settings, get_embeddings(settings))
        app.state.service.ctx.retriever._bm25_count = -1  # forza la ricostruzione dell'indice BM25
        return {"changed": True, **stats}

    def backup() -> dict:
        from pmcopilot.ops.backup import create_backup

        return {"path": str(create_backup(settings))}

    scheduler.add("alerts", 60, alerts)
    scheduler.add("scoring", settings.scoring_interval_min * 60, scoring)
    scheduler.add("drift", settings.drift_interval_min * 60, drift)
    scheduler.add("reindex", 600, reindex)
    scheduler.add("backup", settings.backup_interval_hours * 3600, backup)
    return scheduler
