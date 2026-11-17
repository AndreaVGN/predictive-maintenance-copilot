"""
metrics.py — metriche di esercizio calcolate dallo storico in app.db.

PRIMA ERA IN MEMORIA, ORA È PERSISTENTE
    Un contatore in RAM si azzera a ogni riavvio e vale per un solo processo.
    Ogni richiesta ora è una riga nella tabella `requests` di app.db: le
    metriche si calcolano da lì, per qualunque finestra temporale, anche dopo
    un riavvio, e con più processi uvicorn che scrivono nello stesso file.

DUE FORMATI
    GET /metrics             JSON leggibile (console, script)
    GET /metrics/prometheus  formato testuale di Prometheus: un server Prometheus
                             lo legge ogni 15 secondi ("scrape") e Grafana ne fa
                             grafici e allarmi. È lo standard de facto in produzione.

PERCENTILI, NON MEDIE
    La latenza media nasconde i casi lenti. Il p95 dice: "il 95% delle richieste
    è più veloce di così". È la metrica che conta per l'esperienza utente.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from pmcopilot.store import AppStore


def compute_metrics(store: AppStore, minutes: int | None = None) -> dict[str, Any]:
    """Aggrega le richieste (tutte, o quelle degli ultimi `minutes` minuti)."""
    rows = store.request_stats(minutes)["rows"]
    lat = np.array([r["latency_ms"] or 0 for r in rows], dtype=float) if rows else np.array([0.0])
    routes = Counter(r["route"] for r in rows)
    flags: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    for r in rows:
        flags.update(r["guard_flags"] or [])
        tools.update(r["tools"] or {})
    errors = sum(1 for r in rows if r["error"])
    feedback = store.query("SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating")
    fb = {("positivi" if f["rating"] > 0 else "negativi"): f["n"] for f in feedback}
    return {
        "window_minutes": minutes,
        "requests": len(rows),
        "errors": errors,
        "error_rate": round(errors / len(rows), 4) if rows else 0.0,
        "routes": dict(routes),
        "guard_flags": dict(flags),
        "tool_calls": dict(tools),
        "tokens": {"in": sum(r["tokens_in"] or 0 for r in rows), "out": sum(r["tokens_out"] or 0 for r in rows)},
        "cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 4),
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1),
                       "p95": round(float(np.percentile(lat, 95)), 1),
                       "max": round(float(lat.max()), 1)},
        "feedback": fb,
        "open_alerts": len(store.query("SELECT id FROM alerts WHERE resolved_at IS NULL")),
    }


def daily_series(store: AppStore, days: int = 14) -> list[dict[str, Any]]:
    """Una riga al giorno: richieste, errori, latenza p95, token, costo (per i grafici)."""
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
    rows = store.query("SELECT substr(ts, 1, 10) AS day, latency_ms, error, tokens_in, tokens_out, cost_usd "
                       "FROM requests WHERE ts >= ? ORDER BY ts", (since,))
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["day"], []).append(r)
    out = []
    for day, items in by_day.items():
        lat = [i["latency_ms"] or 0 for i in items]
        out.append({"day": day, "requests": len(items), "errors": sum(1 for i in items if i["error"]),
                    "p95_ms": round(float(np.percentile(lat, 95)), 1),
                    "tokens": sum((i["tokens_in"] or 0) + (i["tokens_out"] or 0) for i in items),
                    "cost_usd": round(sum(i["cost_usd"] or 0 for i in items), 4)})
    return out


def prometheus_text(store: AppStore, extra_gauges: dict[str, float] | None = None) -> str:
    """Formato di esposizione Prometheus: '# HELP', '# TYPE' e una riga per serie."""
    m = compute_metrics(store)
    recent = compute_metrics(store, 15)
    lines = [
        "# HELP pmc_requests_total Richieste di chat e decisione elaborate.",
        "# TYPE pmc_requests_total counter",
    ]
    lines += [f'pmc_requests_total{{route="{route}"}} {n}' for route, n in m["routes"].items()] or \
             ['pmc_requests_total{route="none"} 0']
    lines += [
        "# HELP pmc_errors_total Richieste terminate con errore.",
        "# TYPE pmc_errors_total counter",
        f"pmc_errors_total {m['errors']}",
        "# HELP pmc_tokens_total Token consumati.",
        "# TYPE pmc_tokens_total counter",
        f'pmc_tokens_total{{direction="in"}} {m["tokens"]["in"]}',
        f'pmc_tokens_total{{direction="out"}} {m["tokens"]["out"]}',
        "# HELP pmc_cost_usd_total Costo stimato delle chiamate LLM.",
        "# TYPE pmc_cost_usd_total counter",
        f"pmc_cost_usd_total {m['cost_usd']}",
        "# HELP pmc_guard_flags_total Interventi dei guardrail.",
        "# TYPE pmc_guard_flags_total counter",
    ]
    lines += [f'pmc_guard_flags_total{{flag="{flag}"}} {n}' for flag, n in m["guard_flags"].items()]
    lines += [
        "# HELP pmc_latency_p95_ms Latenza p95 negli ultimi 15 minuti.",
        "# TYPE pmc_latency_p95_ms gauge",
        f"pmc_latency_p95_ms {recent['latency_ms']['p95']}",
        "# HELP pmc_open_alerts Alert aperti.",
        "# TYPE pmc_open_alerts gauge",
        f"pmc_open_alerts {m['open_alerts']}",
    ]
    for name, value in (extra_gauges or {}).items():
        lines += [f"# TYPE {name} gauge", f"{name} {value}"]
    return "\n".join(lines) + "\n"
