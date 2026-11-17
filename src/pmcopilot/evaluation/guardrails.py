"""
guardrails.py (evaluation) — quanto sono efficaci i guardrail?

Due numeri vanno letti INSIEME:
    detection rate       % di attacchi bloccati          (più alto = più sicuro)
    false positive rate  % di domande legittime bloccate (più basso = più usabile)
Un guardrail che blocca tutto ha detection 100% ed è inutile.

Con `with_llm=True` si valuta anche il classificatore LLM di injection
(quello che si attiva con PMC_INJECTION_CLASSIFIER=llm): le regole e il
classificatore insieme dovrebbero prendere le parafrasi che le regex perdono.
È un esempio concreto di trade-off misurabile: più sicurezza, più latenza.
"""

from __future__ import annotations

import time

from pmcopilot.agent.graph import InjectionVerdict
from pmcopilot.agent.guardrails import SQLGuardError, check_input, validate_sql
from pmcopilot.agent.llm import get_chat_model
from pmcopilot.agent.prompts import INJECTION_CLASSIFIER_PROMPT
from pmcopilot.config import Settings
from pmcopilot.data.database import READABLE_TABLES
from pmcopilot.evaluation.common import dataset_path, load_jsonl, run_config
from pmcopilot.store import get_store


def _rates(results: list[dict]) -> tuple[float, float]:
    attacks = [r for r in results if r["expected"] == "block"]
    benign = [r for r in results if r["expected"] == "allow"]
    detection = sum(r["blocked"] for r in attacks) / max(len(attacks), 1)
    false_pos = sum(r["blocked"] for r in benign) / max(len(benign), 1)
    return round(detection, 3), round(false_pos, 3)


def evaluate_guardrails(settings: Settings, with_llm: bool = False, classifier_model=None,
                        label: str | None = None, user: str = "system") -> dict:
    classifier = None
    if with_llm:
        llm = classifier_model or get_chat_model(settings)
        classifier = llm.with_structured_output(InjectionVerdict)

    cases_out = []
    input_results = []
    latencies = []
    for i, case in enumerate(load_jsonl(dataset_path(settings, "input_guard_cases.jsonl"))):
        res = check_input(case["text"], settings.max_input_chars)
        blocked, flags = not res.allowed, list(res.flags)
        if classifier is not None and not blocked:
            t0 = time.perf_counter()
            try:
                verdict = classifier.invoke(INJECTION_CLASSIFIER_PROMPT.format(message=case["text"]))
                if getattr(verdict, "is_attack", False):
                    blocked, flags = True, [*flags, "injection:llm_classifier"]
            except Exception as exc:
                flags.append(f"classifier_error:{type(exc).__name__}")
            latencies.append((time.perf_counter() - t0) * 1000)
        row = {"id": f"in{i + 1:02d}", "layer": "input", **case, "blocked": blocked, "flags": flags,
               "passed": blocked == (case["expected"] == "block")}
        input_results.append(row)
        cases_out.append(row)
    in_det, in_fp = _rates(input_results)

    sql_results = []
    for i, case in enumerate(load_jsonl(dataset_path(settings, "sql_guard_cases.jsonl"))):
        try:
            info, blocked = validate_sql(case["sql"], READABLE_TABLES, settings.sql_max_rows), False
        except SQLGuardError as exc:
            info, blocked = str(exc), True
        row = {"id": f"sql{i + 1:02d}", "layer": "sql", **case, "blocked": blocked, "info": info,
               "passed": blocked == (case["expected"] == "block")}
        sql_results.append(row)
        cases_out.append(row)
    sql_det, sql_fp = _rates(sql_results)

    summary = {"input_cases": len(input_results), "input_detection_rate": in_det,
               "input_false_positive_rate": in_fp, "sql_cases": len(sql_results),
               "sql_detection_rate": sql_det, "sql_false_positive_rate": sql_fp,
               "llm_classifier": with_llm}
    if latencies:
        summary["classifier_latency_ms_avg"] = round(sum(latencies) / len(latencies), 1)
    run_id = get_store(settings.app_db_path).save_eval_run(
        "guardrails", summary, cases_out,
        config=run_config(settings, injection_classifier="llm" if with_llm else "off"),
        label=label or ("guardrail + classificatore LLM" if with_llm else "guardrail a regole"), user=user)
    return {"run_id": run_id, **summary}
