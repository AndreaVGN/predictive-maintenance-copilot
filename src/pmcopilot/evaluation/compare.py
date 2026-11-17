"""
compare.py — confrontare due run e decidere se un cambiamento può andare in produzione.

CONFRONTO (pagina Esperimenti e `pmc eval compare A B`)
    * metriche: valore in A, valore in B, differenza, e se la differenza è un
      miglioramento o un peggioramento (per la latenza "più basso è meglio");
    * configurazione: quali parametri sono cambiati (modello, prompt, retriever...);
    * caso per caso: stessi id nei due run -> regressioni (✓ -> ✗),
      miglioramenti (✗ -> ✓), invariati; per il retrieval anche la posizione.
    * prompt: se la versione è diversa, il diff del testo.

QUALITY GATE (`pmc eval gate`)
    Regole in eval/datasets/quality_gates.json:
      min       soglie minime assolute         (es. pass_rate >= 0,7)
      max       soglie massime assolute        (es. latenza p95 <= 90 s)
      max_drop  calo massimo rispetto a un run di riferimento ("baseline")
    Se una regola fallisce il comando esce con codice 1: in CI la pipeline si
    ferma e il cambiamento non viene rilasciato. È il "test" per i sistemi LLM,
    dove la correttezza non è binaria ma statistica.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

from pmcopilot.config import Settings
from pmcopilot.store import AppStore

# Chiavi di configurazione che cambiano a ogni run e non spiegano differenze di qualità.
IGNORED_CONFIG = {"langfuse_session"}
LOWER_IS_BETTER = ("latency", "false_positive", "cost", "errors", "avg_tokens", "avg_llm_calls")


def _direction(metric: str) -> int:
    return -1 if any(word in metric for word in LOWER_IS_BETTER) else 1


def compare_runs(store: AppStore, run_a: str, run_b: str) -> dict[str, Any]:
    a, b = store.get_eval_run(run_a), store.get_eval_run(run_b)
    if not a or not b:
        raise ValueError("Run non trovato.")
    if a["kind"] != b["kind"]:
        raise ValueError(f"Run di tipo diverso: {a['kind']} vs {b['kind']}.")

    metrics = []
    for key in sorted(set(a["summary"]) | set(b["summary"])):
        va, vb = a["summary"].get(key), b["summary"].get(key)
        row: dict[str, Any] = {"metric": key, "a": va, "b": vb}
        if isinstance(va, int | float) and isinstance(vb, int | float) and not isinstance(va, bool):
            delta = vb - va
            row["delta"] = round(delta, 4)
            row["verdict"] = ("uguale" if abs(delta) < 1e-9 else
                              "meglio" if delta * _direction(key) > 0 else "peggio")
        metrics.append(row)

    config_changes = [{"key": k, "a": a["config"].get(k), "b": b["config"].get(k)}
                      for k in sorted(set(a["config"] or {}) | set(b["config"] or {}))
                      if k not in IGNORED_CONFIG and (a["config"] or {}).get(k) != (b["config"] or {}).get(k)]

    cases_a = {str(c.get("id")): c for c in a["cases"]}
    cases_b = {str(c.get("id")): c for c in b["cases"]}
    cases = []
    counts = {"regressioni": 0, "miglioramenti": 0, "invariati": 0, "solo_in_a": 0, "solo_in_b": 0}
    for cid in list(dict.fromkeys([*cases_a, *cases_b])):
        ca, cb = cases_a.get(cid), cases_b.get(cid)
        if ca is None or cb is None:
            counts["solo_in_b" if ca is None else "solo_in_a"] += 1
            continue
        pa, pb = ca.get("passed"), cb.get("passed")
        change = ("regressione" if pa and not pb else "miglioramento" if pb and not pa else "invariato")
        counts["regressioni" if change == "regressione" else
               "miglioramenti" if change == "miglioramento" else "invariati"] += 1
        cases.append({"id": cid, "question": cb.get("question") or cb.get("text") or cb.get("sql"),
                      "change": change, "a": ca, "b": cb})
    order = {"regressione": 0, "miglioramento": 1, "invariato": 2}
    cases.sort(key=lambda c: order[c["change"]])

    prompt_diff = None
    if a.get("prompt_text") and b.get("prompt_text") and a["prompt_text"] != b["prompt_text"]:
        prompt_diff = "\n".join(difflib.unified_diff(
            a["prompt_text"].splitlines(), b["prompt_text"].splitlines(),
            fromfile=f"prompt {a.get('prompt_version')}", tofile=f"prompt {b.get('prompt_version')}",
            lineterm="", n=1))
    return {"kind": a["kind"],
            "a": {k: a[k] for k in ("id", "ts", "label", "config", "prompt_version")},
            "b": {k: b[k] for k in ("id", "ts", "label", "config", "prompt_version")},
            "metrics": metrics, "config_changes": config_changes, "case_counts": counts,
            "cases": cases, "prompt_diff": prompt_diff}


def load_gates(settings: Settings) -> dict[str, Any]:
    path = settings.eval_dir / "datasets" / "quality_gates.json"
    return json.loads(path.read_text()) if path.exists() else {}


def check_gate(settings: Settings, store: AppStore, kind: str, run_id: str | None = None,
               baseline_id: str | None = None) -> dict[str, Any]:
    """Applica le regole di qualità a un run (default: l'ultimo del tipo)."""
    run = store.get_eval_run(run_id) if run_id else store.latest_eval_run(kind)
    if not run:
        raise ValueError(f"Nessun run di tipo '{kind}'.")
    rules = load_gates(settings).get(kind, {})
    summary = run["summary"]
    failures, checks = [], []
    for metric, threshold in rules.get("min", {}).items():
        value = summary.get(metric)
        ok = value is not None and value >= threshold
        checks.append({"rule": f"{metric} >= {threshold}", "value": value, "ok": ok})
    for metric, threshold in rules.get("max", {}).items():
        value = summary.get(metric)
        ok = value is None or value <= threshold
        checks.append({"rule": f"{metric} <= {threshold}", "value": value, "ok": ok})
    if baseline_id and rules.get("max_drop"):
        base = store.get_eval_run(baseline_id)
        for metric, max_drop in rules["max_drop"].items():
            before, after = (base["summary"].get(metric) if base else None), summary.get(metric)
            if before is None or after is None:
                continue
            drop = (before - after) * _direction(metric)
            checks.append({"rule": f"calo di {metric} <= {max_drop} (baseline {baseline_id})",
                           "value": round(drop, 4), "ok": drop <= max_drop})
    failures = [c for c in checks if not c["ok"]]
    return {"run_id": run["id"], "kind": kind, "passed": not failures, "checks": checks}
