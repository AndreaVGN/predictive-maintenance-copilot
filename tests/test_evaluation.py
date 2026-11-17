"""Test della suite di valutazione: deve girare e produrre metriche coerenti."""

from fakes import ScriptedChatModel
from langchain_core.messages import AIMessage

from pmcopilot.agent.service import CopilotService
from pmcopilot.evaluation.agent import JudgeVerdict, evaluate_agent
from pmcopilot.evaluation.common import load_jsonl, write_markdown_report
from pmcopilot.evaluation.guardrails import evaluate_guardrails
from pmcopilot.evaluation.retrieval import evaluate_retrieval


def test_golden_sets_are_valid(env):
    for name in ["retrieval_golden", "input_guard_cases", "sql_guard_cases", "agent_golden"]:
        rows = load_jsonl(env.eval_dir / "datasets" / f"{name}.jsonl")
        assert len(rows) >= 10
    manuals = {p.name for p in env.manuals_dir.glob("*.md")}
    for row in load_jsonl(env.eval_dir / "datasets" / "retrieval_golden.jsonl"):
        assert row["expected_source"] in manuals  # nessun refuso nei nomi dei file


def test_guardrail_eval(env):
    summary = evaluate_guardrails(env)
    assert summary["sql_detection_rate"] == 1.0
    assert summary["input_false_positive_rate"] == 0.0


def test_retrieval_eval_runs(env):
    summary = evaluate_retrieval(env)
    assert 0 <= summary["mrr"] <= 1  # con embedding finti il valore non è significativo


def test_agent_eval_runs_with_fake_models(tool_ctx):
    llm = ScriptedChatModel(script=[AIMessage(content="Risposta generica.")] * 50)
    judge = ScriptedChatModel(script=[], structured=JudgeVerdict(score=4, reasoning="ok"))
    service = CopilotService(tool_ctx.settings, tool_ctx, llm)
    summary = evaluate_agent(tool_ctx.settings, service=service, judge_model=judge, limit=3)
    assert summary["n_cases"] == 3
    report = write_markdown_report(tool_ctx.settings)
    assert "Agente end-to-end" in report.read_text()
    from pmcopilot.store import get_store

    run = get_store(tool_ctx.settings.app_db_path).get_eval_run(summary["run_id"])
    assert len(run["cases"]) == 3 and run["prompt_text"] and run["config"]["llm"]
