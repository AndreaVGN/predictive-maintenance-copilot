"""
Smoke test: girano contro un'istanza VERA già avviata (dopo un deploy).

    PMC_SMOKE_URL=http://localhost:8000 PMC_SMOKE_PASSWORD=... pytest -m smoke tests/smoke

Non usano LLM finti: verificano che il servizio risponda, che l'autenticazione
funzioni, che modello ML e indice siano caricati e (opzionale, più lento) che
l'agente risponda a una domanda vera. Esclusi dall'esecuzione normale dei test
(addopts in pyproject: -m 'not smoke').
"""

import os

import pytest
import requests

URL = os.environ.get("PMC_SMOKE_URL", "").rstrip("/")
USER = os.environ.get("PMC_SMOKE_USER", "admin")
PASSWORD = os.environ.get("PMC_SMOKE_PASSWORD", "admin")

pytestmark = [pytest.mark.smoke, pytest.mark.skipif(not URL, reason="PMC_SMOKE_URL non impostato")]


@pytest.fixture(scope="module")
def session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # niente proxy di sistema verso localhost
    resp = s.post(f"{URL}/auth/login", json={"username": USER, "password": PASSWORD}, timeout=10)
    assert resp.status_code == 200, resp.text
    return s


def test_liveness_and_readiness():
    assert requests.get(f"{URL}/health/live", timeout=5).json()["status"] == "ok"
    ready = requests.get(f"{URL}/health/ready", timeout=10)
    assert ready.status_code == 200, ready.json()


def test_protected_without_login():
    assert requests.get(f"{URL}/config", timeout=5).status_code == 401


def test_prediction(session):
    body = {"machine_type": "L", "air_temp_k": 300, "process_temp_k": 310, "rpm": 1500,
            "torque_nm": 40, "tool_wear_min": 50}
    resp = session.post(f"{URL}/predict", json=body, timeout=10)
    assert resp.status_code == 200 and 0 <= resp.json()["failure_proba"] <= 1


def test_config_and_machines(session):
    cfg = session.get(f"{URL}/config", timeout=10).json()
    assert cfg["model_version"] and cfg["llm_model"]
    assert session.get(f"{URL}/machines", timeout=10).json()


@pytest.mark.skipif(not os.environ.get("PMC_SMOKE_CHAT"), reason="imposta PMC_SMOKE_CHAT=1 per provare l'LLM")
def test_chat_round_trip(session):
    resp = session.post(f"{URL}/chat", json={"message": "Quante macchine ci sono per ogni linea?"}, timeout=300)
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "in_scope" and body["answer"]
