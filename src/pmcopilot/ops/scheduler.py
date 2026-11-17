"""
scheduler.py — i lavori periodici che un sistema in produzione fa da solo.

I LAVORI
    scoring        ogni ora: il modello valuta le letture recenti (tabella predictions)
    drift          ogni ora: confronto della finestra recente con il training
    alerts         ogni minuto: valutazione delle regole di allarme
    reindex        ogni 10 minuti: se i manuali sono cambiati, reindicizzazione incrementale
    backup         ogni giorno: copia consistente di database e modello

COME È FATTO
    Un thread in background avviato dall'API (lifespan). Ogni lavoro ha un
    intervallo; il thread si sveglia ogni 5 secondi e lancia quelli scaduti.
    Ogni esecuzione è isolata da try/except: se il drift fallisce, gli alert
    continuano a girare. Lo stato (ultima esecuzione, durata, esito) è visibile
    in GET /ops/jobs e nella pagina Monitoraggio.

E IN UN'AZIENDA VERA?
    Con più istanze dell'API, ognuna avrebbe il suo scheduler e i lavori
    girerebbero N volte. Le alternative: un solo "worker" dedicato, un cron di
    sistema che chiama `pmc score` / `pmc drift check`, oppure un orchestratore
    (Airflow, Prefect, Kubernetes CronJob). Il codice dei lavori resta lo stesso:
    cambia solo CHI li chiama. Per questo ogni lavoro è anche un comando `pmc`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pmcopilot.store import now_iso

logger = logging.getLogger(__name__)


@dataclass
class Job:
    name: str
    interval_s: float
    func: Callable[[], Any]
    run_at_start: bool = False
    last_run: str | None = None
    last_duration_ms: float | None = None
    last_status: str = "mai eseguito"
    last_result: Any = None
    runs: int = 0
    _next: float = field(default=0.0, repr=False)


class Scheduler:
    def __init__(self, tick_s: float = 5.0):
        self.jobs: dict[str, Job] = {}
        self.tick_s = tick_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def add(self, name: str, interval_s: float, func: Callable[[], Any], run_at_start: bool = False) -> None:
        job = Job(name, interval_s, func, run_at_start)
        job._next = time.monotonic() + (0 if run_at_start else interval_s)
        self.jobs[name] = job

    def run_job(self, name: str) -> Job:
        """Esegue subito un lavoro (anche dalla UI: "esegui ora")."""
        job = self.jobs[name]
        with self._lock:
            start = time.perf_counter()
            try:
                job.last_result = job.func()
                job.last_status = "ok"
            except Exception as exc:
                job.last_status = f"errore: {type(exc).__name__}: {exc}"[:300]
                logger.exception("Lavoro '%s' fallito", name)
            job.last_duration_ms = round((time.perf_counter() - start) * 1000, 1)
            job.last_run = now_iso()
            job.runs += 1
            job._next = time.monotonic() + job.interval_s
        return job

    def _loop(self) -> None:
        while not self._stop.wait(self.tick_s):
            for name, job in list(self.jobs.items()):
                if time.monotonic() >= job._next:
                    self.run_job(name)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="pmc-scheduler", daemon=True)
            self._thread.start()
            logger.info("Scheduler avviato con %d lavori: %s", len(self.jobs), ", ".join(self.jobs))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def status(self) -> list[dict[str, Any]]:
        return [{"name": j.name, "interval_s": j.interval_s, "last_run": j.last_run,
                 "last_duration_ms": j.last_duration_ms, "last_status": j.last_status, "runs": j.runs,
                 "next_in_s": round(max(0.0, j._next - time.monotonic()))} for j in self.jobs.values()]
