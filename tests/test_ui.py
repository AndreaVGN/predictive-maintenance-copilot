"""Test della console web: pagina, endpoint dati e streaming SSE dei passi dell'agente."""

import json

import pytest
from conftest import login
from fakes import ScriptedChatModel, tool_call
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from pmcopilot.agent.service import CopilotService
from pmcopilot.api.main import create_app
from pmcopilot.observability.langfuse_setup import _read_env, init_local_langfuse


def _client(tool_ctx, script):
    service = CopilotService(tool_ctx.settings, tool_ctx, ScriptedChatModel(script=script))
    return TestClient(create_app(settings=tool_ctx.settings, service=service, start_scheduler=False))


def _events(response) -> list[dict]:
    """Estrae gli eventi da una risposta Server-Sent Events."""
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def client(tool_ctx):
    with _client(tool_ctx, [AIMessage(content="ok")]) as c:
        login(c)
        yield c


def test_index_and_static_files(client):
    page = client.get("/")
    assert page.status_code == 200 and "Copilota manutenzione" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_config_machines_and_detail(client):
    cfg = client.get("/config").json()
    assert cfg["llm_model"] and cfg["langfuse"]["enabled"] is False
    assert cfg["context_window"] == 8192
    assert cfg["user"] == {"name": "admin", "role": "admin"}
    machines = client.get("/machines").json()
    assert machines and {"machine_id", "risk_level", "n_failures", "tool_wear_min"} <= machines[0].keys()
    detail = client.get(f"/machines/{machines[0]['machine_id']}?hours=24").json()
    assert detail["readings"] and "failure_proba" in detail["readings"][0]
    assert client.get("/machines/M99").status_code == 404


def test_model_eval_and_figures(client):
    model = client.get("/model").json()
    assert model["card"]["threshold"] > 0 and model["figures"]
    assert client.get(f"/reports/figures/{model['figures'][0]}").status_code == 200
    assert client.get("/reports/figures/..%2Fmodel_report.md").status_code in (400, 404)
    task = client.post("/experiments/run", json={"kind": "guardrails"}).json()
    import time

    for _ in range(50):
        state = client.get(f"/ops/tasks/{task['id']}").json()
        if state["status"] in ("completato", "errore"):
            break
        time.sleep(0.1)
    assert state["status"] == "completato", state
    assert state["result"]["sql_detection_rate"] == 1.0
    runs = client.get("/experiments/runs", params={"kind": "guardrails"}).json()
    assert runs and runs[0]["kind"] == "guardrails"


def test_chat_stream_emits_steps_in_order(tool_ctx):
    script = [tool_call("sql_query", {"query": "SELECT COUNT(*) AS n FROM machines"}),
              AIMessage(content="Ci sono 6 macchine.")]
    with _client(tool_ctx, script) as c:
        login(c)
        resp = c.post("/chat/stream", json={"message": "Quante macchine?"})
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _events(resp)
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "final"
    assert types.index("tool_call") < types.index("tool_result") < types.index("final")
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["tool"] == "sql_query" and "SELECT" in call["args"]["query"]
    final = events[-1]["result"]
    assert final["answer"] == "Ci sono 6 macchine." and "tool_outputs" not in final
    assert "context_peak" in final["trace"]  # usato dal contatore del contesto nella console


def test_stream_approval_flow(tool_ctx):
    args = {"machine_id": "M01", "failure_type": "TWF", "priority": "P3", "notes": "utensile"}
    with _client(tool_ctx, [tool_call("open_work_order", args), AIMessage(content="Registrato.")]) as c:
        login(c)
        first = _events(c.post("/chat/stream", json={"message": "Apri un ordine"}))[-1]["result"]
        assert first["pending_approval"][0]["tool"] == "open_work_order"
        events = _events(c.post(f"/chat/{first['thread_id']}/decision/stream", json={"approve": True}))
        assert events[-1]["result"]["answer"] == "Registrato."
        assert any(e["type"] == "tool_result" and e["tool"] == "open_work_order" for e in events)
        assert c.post(f"/chat/{first['thread_id']}/decision/stream", json={"approve": True}).status_code == 404


def test_langfuse_init_is_idempotent(tmp_path):
    (tmp_path / ".env.example").write_text("PMC_LANGFUSE_ENABLED=false\nLANGFUSE_PUBLIC_KEY=pk-lf-...\n")
    first = init_local_langfuse(tmp_path)
    second = init_local_langfuse(tmp_path)
    assert first["password"] == second["password"]  # non rigenera i segreti
    app_env = _read_env(tmp_path / ".env")
    infra_env = _read_env(tmp_path / "infra" / "langfuse" / ".env")
    assert app_env["PMC_LANGFUSE_ENABLED"] == "true"
    assert app_env["LANGFUSE_PUBLIC_KEY"] == infra_env["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"]
    assert app_env["LANGFUSE_SECRET_KEY"] == infra_env["LANGFUSE_INIT_PROJECT_SECRET_KEY"]
    assert len(infra_env["ENCRYPTION_KEY"]) == 64
    assert (tmp_path / ".env").read_text().count("LANGFUSE_PUBLIC_KEY=") == 1


def test_experiments_and_monitoring_endpoints(client):
    """Le API usate dalle pagine Esperimenti e Monitoraggio."""
    import time

    ids = []
    for _ in range(2):
        task = client.post("/experiments/run", json={"kind": "retrieval", "mode": "bm25", "label": "test"}).json()
        for _ in range(100):
            state = client.get(f"/ops/tasks/{task['id']}").json()
            if state["status"] in ("completato", "errore"):
                break
            time.sleep(0.1)
        assert state["status"] == "completato", state
        ids.append(state["result"][0]["run_id"])
    cmp = client.get("/experiments/compare", params={"a": ids[0], "b": ids[1]}).json()
    assert cmp["case_counts"]["regressioni"] == 0  # stessa configurazione, stessi risultati (BM25 è deterministico)
    assert client.get("/experiments/compare", params={"a": ids[0], "b": "inesistente"}).status_code == 400
    trend = client.get("/experiments/trend", params={"kind": "retrieval"}).json()
    assert trend[-1]["id"] == ids[1] and trend[-1]["retriever_mode"] == "bm25"
    gate = client.get("/experiments/gate", params={"kind": "retrieval", "run": ids[1], "baseline": ids[0]}).json()
    assert {c["rule"].split()[0] for c in gate["checks"]} >= {"hit@4", "mrr"}
    assert client.get(f"/experiments/runs/{ids[0]}").json()["cases"]

    jobs = {j["name"] for j in client.get("/ops/jobs").json()}
    assert {"scoring", "drift", "alerts", "reindex", "backup"} <= jobs
    assert client.post("/ops/jobs/alerts/run").json()["status"] == "ok"
    assert client.post("/ops/jobs/nessuno/run").status_code == 404
    assert isinstance(client.get("/ops/alerts").json(), list)
    assert "pmc_requests_total" in client.get("/metrics/prometheus").text
    assert any(a["action"] == "run_job" for a in client.get("/ops/audit").json())
    assert client.get("/ops/models").json()[0]["status"] == "production"
