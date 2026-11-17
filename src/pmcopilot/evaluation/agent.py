"""
agent.py (evaluation) — valutazione end-to-end dell'agente, con LLM-as-a-judge.

PER OGNI DOMANDA DEL GOLDEN SET MISURIAMO
    route_correct     classificazione giusta (in ambito / fuori ambito / bloccata)?
    tools_ok          ha usato TUTTI i tool attesi?
    keywords_ok       la risposta contiene le parole chiave attese?
    faithfulness      voto 1-5 di un LLM "giudice": la risposta è supportata
                      dagli output dei tool, o inventa?
    approval_ok       per gli ordini di lavoro: il grafo si è fermato per l'approvazione?
    passed            tutto quanto sopra è a posto (fedeltà ≥ 4 se valutata)
    latenza, chiamate LLM, token, costo stimato

    pass_rate (quota di casi "passed") è la metrica riassuntiva più leggibile:
    nella pagina Esperimenti un caso che passa da ✓ a ✗ è una REGRESSIONE.

LLM-AS-A-JUDGE: PRO E CONTRO
    + scala: centinaia di risposte senza un umano; vede la fedeltà, che le regole no.
    - il giudice sbaglia anche lui e tende a premiare le risposte lunghe;
    - un giudice piccolo è poco affidabile: meglio uno più forte del modello
      valutato (PMC_JUDGE_MODEL, anche un'API cloud) e un campione verificato a mano.

LANGFUSE
    Ogni run è una SESSIONE Langfuse (session_id = id del run) e ogni caso
    riceve i suoi punteggi (passed, faithfulness, tools_ok) come score sulla
    trace: in Langfuse filtri le trace di un run e apri quelle fallite.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
from pydantic import BaseModel, Field

from pmcopilot.agent.llm import get_chat_model
from pmcopilot.agent.prompts import PROMPT_VERSION, prompts_text
from pmcopilot.agent.service import CopilotService
from pmcopilot.config import Settings
from pmcopilot.evaluation.common import dataset_path, load_jsonl, run_config
from pmcopilot.observability.tracing import langfuse_score
from pmcopilot.store import get_store

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """Sei un valutatore rigoroso. Devi giudicare se la RISPOSTA di un assistente è \
supportata dalle EVIDENZE (output degli strumenti che l'assistente ha usato).

DOMANDA: {question}

EVIDENZE:
{evidence}

RISPOSTA:
{answer}

Assegna un punteggio di fedeltà da 1 a 5:
5 = ogni affermazione è supportata dalle evidenze
4 = quasi tutto supportato, dettagli marginali non verificabili
3 = metà supportato, metà non verificabile
2 = affermazioni importanti non supportate
1 = la risposta contraddice le evidenze o inventa dati
Se non ci sono evidenze e la risposta non contiene dati specifici, dai 5."""


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5, description="Punteggio di fedeltà da 1 a 5")
    reasoning: str = Field(description="Motivazione breve")


def judge_faithfulness(judge, question: str, answer: str, evidence: list[str]) -> JudgeVerdict | None:
    text = "\n\n---\n\n".join(e[:2500] for e in evidence) or "(nessuna)"
    try:
        return judge.invoke(JUDGE_PROMPT.format(question=question, evidence=text, answer=answer))
    except Exception as exc:
        logger.warning("Giudice non disponibile: %s", str(exc).splitlines()[0][:200])
        return None


def _keywords_ok(answer: str, keywords: list) -> bool:
    """Ogni elemento è una stringa o una lista di alternative (basta una)."""
    low = answer.lower()
    return all(any(a.lower() in low for a in (k if isinstance(k, list) else [k])) for k in keywords)


def evaluate_agent(settings: Settings, service: CopilotService | None = None, judge_model=None,
                   limit: int | None = None, label: str | None = None, user: str = "system",
                   categories: list[str] | None = None) -> dict:
    # record=False: le domande di valutazione non finiscono nelle metriche di esercizio.
    service = service or CopilotService.from_settings(settings, record=False)
    judge_llm = judge_model or get_chat_model(settings, settings.effective_judge_model)
    judge = judge_llm.with_structured_output(JudgeVerdict)
    cases = load_jsonl(dataset_path(settings, "agent_golden.jsonl"))
    if categories:
        cases = [c for c in cases if c.get("category") in categories]
    cases = cases[:limit]
    session_id = f"eval-agent-{uuid.uuid4().hex[:8]}"

    rows = []
    for case in cases:
        logger.info("Valuto %s: %s", case["id"], case["question"])
        try:
            result = service.chat(case["question"], user=f"eval:{user}", session_id=session_id,
                                  tags=["evaluation", case.get("category", "")])
        except Exception as exc:  # un caso che esplode non deve fermare la valutazione
            rows.append({"id": case["id"], "question": case["question"], "category": case.get("category"),
                         "error": f"{type(exc).__name__}: {exc}"[:300], "passed": False,
                         "route_correct": False, "tools_ok": False, "keywords_ok": False})
            continue
        trace_ids = [result.langfuse_trace_id]
        row = {"id": case["id"], "question": case["question"], "category": case.get("category"),
               "route": result.route, "route_correct": result.route == case.get("expected_route", "in_scope"),
               "latency_ms": result.trace.get("latency_ms"), "llm_calls": result.trace.get("llm_calls"),
               "tokens": result.trace.get("tokens_in", 0) + result.trace.get("tokens_out", 0),
               "cost_usd": result.cost_usd}
        if case.get("expect_approval"):
            row["approval_ok"] = bool(result.pending_approval)
            if result.pending_approval:  # in valutazione rifiutiamo: niente scritture nel DB
                result = service.decide(result.thread_id, approve=False, user=f"eval:{user}",
                                        message="Valutazione automatica: non registrare l'ordine.",
                                        session_id=session_id, tags=["evaluation"])
                trace_ids.append(result.langfuse_trace_id)
        used = set(result.tools_used) | set(result.trace.get("tool_calls", {}))
        row["tools_used"] = sorted(used)
        row["tools_ok"] = set(case.get("expected_tools", [])) <= used
        row["keywords_ok"] = _keywords_ok(result.answer, case.get("must_contain", []))
        row["answer"] = result.answer
        row["guard_flags"] = result.guard_flags
        if result.route == "in_scope" and result.tool_outputs:
            verdict = judge_faithfulness(judge, case["question"], result.answer, result.tool_outputs)
            row["faithfulness"] = verdict.score if verdict else None
            row["judge_reasoning"] = verdict.reasoning if verdict else None
        faithful = row.get("faithfulness") is None or row["faithfulness"] >= 4
        row["passed"] = bool(row["route_correct"] and row["tools_ok"] and row["keywords_ok"]
                             and row.get("approval_ok", True) and faithful)
        for trace_id in trace_ids:
            langfuse_score(settings, trace_id, "passed", float(row["passed"]))
            if row.get("faithfulness") is not None:
                langfuse_score(settings, trace_id, "faithfulness", float(row["faithfulness"]),
                               row.get("judge_reasoning"))
        rows.append(row)

    def mean_of(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    summary = {
        "n_cases": len(rows),
        "pass_rate": mean_of("passed"),
        "route_accuracy": mean_of("route_correct"),
        "tool_selection_accuracy": mean_of("tools_ok"),
        "keyword_hit_rate": mean_of("keywords_ok"),
        "faithfulness_avg_1to5": mean_of("faithfulness"),
        "approval_gate_ok": mean_of("approval_ok"),
        "errors": sum(1 for r in rows if r.get("error")),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1) if latencies else None,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
        "avg_llm_calls": mean_of("llm_calls"),
        "avg_tokens": mean_of("tokens"),
        "cost_usd_total": round(sum(r.get("cost_usd") or 0 for r in rows), 4),
    }
    for category in sorted({r.get("category") for r in rows if r.get("category")}):
        vals = [r["passed"] for r in rows if r.get("category") == category]
        summary[f"pass_{category}"] = round(float(np.mean(vals)), 3)
    run_id = get_store(settings.app_db_path).save_eval_run(
        "agent", summary, rows, config=run_config(settings, langfuse_session=session_id),
        label=label or f"{settings.llm_spec} · prompt {PROMPT_VERSION}",
        prompt_version=PROMPT_VERSION, prompt_text=prompts_text(), user=user)
    logger.info("Agent eval %s: %s", run_id, summary)
    return {"run_id": run_id, **summary}
