"""Test dell'API FastAPI con TestClient (nessun server reale, nessun Ollama)."""

import pytest
from conftest import login
from fakes import ScriptedChatModel, tool_call
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from pmcopilot.agent.service import CopilotService
from pmcopilot.api.main import create_app


def _client(tool_ctx, script):
    service = CopilotService(tool_ctx.settings, tool_ctx, ScriptedChatModel(script=script))
    return TestClient(create_app(settings=tool_ctx.settings, service=service, start_scheduler=False))


@pytest.fixture
def client(tool_ctx):
    script = [
        tool_call("make_chart", {"machine_id": "M01", "metric": "rpm", "last_hours": 24}),
        AIMessage(content="Ecco il grafico dei giri della M01."),
    ]
    with _client(tool_ctx, script) as c:  # "with" esegue il lifespan
        login(c)
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["checks"]["database"] and body["checks"]["model"]


def test_predict_valid_and_invalid(client):
    ok = client.post("/predict", json={"machine_type": "L", "air_temp_k": 300, "process_temp_k": 310,
                                       "rpm": 1500, "torque_nm": 40, "tool_wear_min": 50})
    assert ok.status_code == 200
    assert 0 <= ok.json()["failure_proba"] <= 1
    bad = client.post("/predict", json={"machine_type": "X", "air_temp_k": 5000, "process_temp_k": 310,
                                        "rpm": 1500, "torque_nm": 40, "tool_wear_min": 50})
    assert bad.status_code == 422  # Pydantic rifiuta l'input


def test_chat_returns_chart_and_request_id(client):
    resp = client.post("/chat", json={"message": "Grafico dei giri della M01"})
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    body = resp.json()
    assert body["charts"] and body["charts"][0].startswith("/charts/")
    png = client.get(body["charts"][0])
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert client.get("/metrics").json()["requests"] >= 1


def test_chart_path_traversal_blocked(client):
    assert client.get("/charts/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)


def test_chat_approval_flow(tool_ctx):
    args = {"machine_id": "M02", "failure_type": "OSF", "priority": "P3", "notes": "controllo"}
    script = [tool_call("open_work_order", args), AIMessage(content="Ordine registrato.")]
    with _client(tool_ctx, script) as c:
        login(c)
        first = c.post("/chat", json={"message": "Apri un ordine per M02"}).json()
        assert first["pending_approval"]
        tid = first["thread_id"]
        # Mentre c'è un'approvazione in sospeso non si può continuare a chattare.
        assert c.post("/chat", json={"message": "e poi?", "thread_id": tid}).status_code == 409
        done = c.post(f"/chat/{tid}/decision", json={"approve": True}).json()
        assert done["answer"] == "Ordine registrato."
        assert c.post(f"/chat/{tid}/decision", json={"approve": True}).status_code == 404
