"""
logging.py — log strutturati in JSON con un identificativo di richiesta.

PERCHÉ JSON
    Un log testuale ("ERRORE alle 10:32 nella chat") va bene per un umano;
    un log JSON ({"level": "ERROR", "request_id": "...", "latency_ms": 812})
    si può filtrare, aggregare e caricare in strumenti come Grafana Loki,
    Elasticsearch o anche solo pandas. È lo standard in produzione.

IL REQUEST_ID
    Quando arriva una richiesta HTTP, l'API genera un id univoco e lo mette in
    una ContextVar. Ogni riga di log scritta DURANTE quella richiesta, da
    qualunque modulo, lo include automaticamente. Così puoi ricostruire tutto
    ciò che è successo per una singola domanda dell'utente.
    ContextVar (invece di una variabile globale) funziona anche con thread e
    asyncio: ogni richiesta vede il proprio valore.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pmcopilot.agent.guardrails import redact_pii

# Valore di default "-" quando non siamo dentro una richiesta (es. script CLI).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """Trasforma ogni LogRecord in una riga JSON (con i dati personali oscurati)."""

    def __init__(self, redact: bool = True):
        super().__init__()
        self.redact = redact

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        # Campi extra passati con logger.info("...", extra={"fields": {...}})
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        return redact_pii(line) if self.redact else line


def setup_logging(level: str = "INFO", json_format: bool = True, log_file: Path | None = None,
                  max_mb: int = 20, backups: int = 5, redact: bool = True) -> None:
    """Configura il logger radice. Da chiamare una volta all'avvio (CLI o API).

    Due destinazioni:
      * stderr: per il terminale e per i container (Docker raccoglie stderr);
      * logs/app.log (se log_file): file JSON con ROTAZIONE. RotatingFileHandler
        apre un nuovo file quando si supera max_mb e tiene solo `backups` file:
        lo spazio su disco resta limitato (retention).
    """
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(JsonFormatter(redact) if json_format else logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    handlers: list[logging.Handler] = [console]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=max_mb * 1024 * 1024,
                                           backupCount=backups, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter(redact))  # su file sempre JSON
        handlers.append(file_handler)
    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "matplotlib", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def setup_logging_from_settings(settings, json_format: bool = True, level: str | None = None) -> None:
    setup_logging(level or settings.log_level, json_format,
                  settings.app_log_path if settings.log_to_file else None,
                  settings.log_max_mb, settings.log_backups, settings.redact_pii)
