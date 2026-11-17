"""
alerts.py — regole di allarme: chi ti sveglia (o ti scrive) quando qualcosa va storto.

COME FUNZIONA
    Ogni minuto lo scheduler valuta alcune REGOLE sui dati di esercizio.
    Se una regola scatta, si apre un alert (uno solo per regola finché resta
    aperto, così non arrivano 60 notifiche in un'ora). Quando la condizione
    rientra, l'alert si chiude da solo. Apertura e chiusura vengono:
      * scritte nel log (livello WARNING / INFO);
      * salvate in app.db (storico, visibile nella pagina Monitoraggio);
      * inviate a un webhook, se configurato (Slack, Teams, ...).

LE REGOLE
    error_rate        troppe richieste fallite negli ultimi N minuti
    latency_p95       le risposte sono diventate lente
    guard_spike       molte richieste bloccate: attacco in corso o regole troppo severe?
    negative_feedback gli utenti bocciano le risposte
    data_drift        l'ultimo controllo del drift è in stato "alert"
    ollama_down       il server del modello non risponde

    Le soglie stanno nel .env (PMC_ALERT_*). Una regola richiede un minimo di
    richieste prima di scattare: con 2 richieste, 1 errore non è "il 50%".
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from pmcopilot.config import Settings
from pmcopilot.observability.metrics import compute_metrics
from pmcopilot.store import AppStore

logger = logging.getLogger(__name__)
MIN_REQUESTS = 5


def _post_webhook(url: str, text: str) -> None:
    body = json.dumps({"text": text}).encode()  # formato accettato da Slack e Teams
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)  # noqa: S310
    except Exception as exc:
        logger.warning("Webhook alert non raggiungibile: %s", exc)


def evaluate_rules(settings: Settings, store: AppStore, ollama_ok: bool | None = None) -> list[dict[str, Any]]:
    """Valuta tutte le regole e restituisce lo stato di ciascuna."""
    window = settings.alert_window_min
    m = compute_metrics(store, window)
    checks: list[tuple[str, str, bool, str]] = []
    enough = m["requests"] >= MIN_REQUESTS

    checks.append(("error_rate", "critical", enough and m["error_rate"] > settings.alert_error_rate,
                   f"{m['error_rate']:.0%} di richieste in errore negli ultimi {window} minuti "
                   f"(soglia {settings.alert_error_rate:.0%})."))
    checks.append(("latency_p95", "warning",
                   enough and m["latency_ms"]["p95"] > settings.alert_p95_latency_s * 1000,
                   f"Latenza p95 {m['latency_ms']['p95'] / 1000:.1f} s negli ultimi {window} minuti "
                   f"(soglia {settings.alert_p95_latency_s:.0f} s)."))
    blocked = m["routes"].get("blocked", 0)
    checks.append(("guard_spike", "warning", enough and blocked / max(m["requests"], 1) > 0.3,
                   f"{blocked} richieste su {m['requests']} bloccate dai guardrail negli ultimi {window} minuti."))
    fb = store.query("SELECT rating FROM feedback ORDER BY ts DESC LIMIT 20")
    negatives = sum(1 for f in fb if f["rating"] < 0)
    checks.append(("negative_feedback", "warning", len(fb) >= MIN_REQUESTS and negatives / len(fb) > 0.3,
                   f"{negatives} degli ultimi {len(fb)} feedback sono negativi."))
    drift = store.drift_reports(limit=1)
    checks.append(("data_drift", "warning", bool(drift) and drift[0]["status"] == "alert",
                   f"Drift rilevato: {', '.join(drift[0]['prediction'].get('diagnosis', [])) if drift else ''} "
                   f"(controllo del {drift[0]['ts'] if drift else '-'})."))
    if ollama_ok is not None:
        checks.append(("ollama_down", "critical", not ollama_ok,
                       f"Ollama non risponde su {settings.ollama_base_url}."))

    results = []
    for rule, severity, firing, message in checks:
        if firing and store.open_alert(rule, severity, message):
            logger.warning("ALERT aperto [%s] %s", rule, message)
            if settings.alert_webhook_url:
                _post_webhook(settings.alert_webhook_url, f":rotating_light: [{severity}] {message}")
        elif not firing and store.resolve_alert(rule):
            logger.info("Alert risolto [%s]", rule)
            if settings.alert_webhook_url:
                _post_webhook(settings.alert_webhook_url, f":white_check_mark: risolto: {rule}")
        results.append({"rule": rule, "severity": severity, "firing": firing, "message": message})
    return results
