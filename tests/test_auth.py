"""
Test di autenticazione, ruoli e limiti dell'API.

Cosa deve valere in produzione:
  * senza credenziali nessun endpoint di dati risponde (401);
  * il cookie di sessione è firmato: se lo modifichi, non vale più;
  * l'operatore chatta ma NON approva azioni e non vede le pagine admin (403);
  * dopo 5 password sbagliate lo stesso client viene bloccato (429);
  * rate limit e budget giornaliero di token fermano gli abusi (429);
  * in ambiente "prod" l'API non parte con la password di default.
"""

import base64

import pytest
from conftest import login
from fakes import ScriptedChatModel, tool_call
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from pmcopilot.agent.service import CopilotService
from pmcopilot.api.auth import COOKIE_NAME, Authenticator, User, verify_credentials
from pmcopilot.api.main import create_app
from pmcopilot.store import get_store


def _app(tool_ctx, settings=None, script=None):
    settings = settings or tool_ctx.settings
    service = CopilotService(settings, tool_ctx, ScriptedChatModel(script=script or [AIMessage(content="ok")]))
    return TestClient(create_app(settings=settings, service=service, start_scheduler=False))


@pytest.fixture
def with_operator(tool_ctx):
    """Le stesse impostazioni dei test, più un utente operatore."""
    return tool_ctx.settings.model_copy(update={"operator_user": "operatore", "operator_password": SecretStr("op-pass")})


def test_endpoints_require_authentication(tool_ctx):
    with _app(tool_ctx) as c:
        assert c.get("/health/live").status_code == 200  # la sonda di liveness resta pubblica
        for url in ["/health", "/config", "/machines", "/metrics", "/experiments/runs"]:
            assert c.get(url).status_code == 401, url
        assert c.post("/chat", json={"message": "ciao"}).status_code == 401
        assert c.get("/").status_code == 200  # la pagina si carica e mostra il login


def test_login_wrong_password_and_logout(tool_ctx):
    with _app(tool_ctx) as c:
        bad = c.post("/auth/login", json={"username": "admin", "password": "no"})
        assert bad.status_code == 401
        ok = c.post("/auth/login", json={"username": "admin", "password": "admin"})
        assert ok.json() == {"user": "admin", "role": "admin", "default_password": True}
        assert c.get("/auth/me").json()["role"] == "admin"
        c.post("/auth/logout")
        assert c.get("/auth/me").status_code == 401


def test_login_lockout_after_repeated_failures(tool_ctx):
    with _app(tool_ctx) as c:
        for _ in range(5):
            assert c.post("/auth/login", json={"username": "admin", "password": "x"}).status_code == 401
        # Dal sesto tentativo il client è bloccato, anche con la password giusta.
        assert c.post("/auth/login", json={"username": "admin", "password": "admin"}).status_code == 429


def test_tampered_cookie_is_rejected(tool_ctx):
    auth = Authenticator(tool_ctx.settings)
    cookie = auth.make_cookie(User("operatore", "operatore"))
    assert auth.read_cookie(cookie).role == "operatore"
    payload, rest = cookie.split(".", 1)
    forged = base64.urlsafe_b64encode(b'{"u":"operatore","r":"admin"}').decode().rstrip("=")
    assert auth.read_cookie(f"{forged}.{rest}") is None  # la firma non torna più
    with _app(tool_ctx) as c:
        c.cookies.set(COOKIE_NAME, f"{forged}.{rest}")
        assert c.get("/config").status_code == 401


def test_basic_auth_for_scripts(tool_ctx):
    with _app(tool_ctx) as c:
        token = base64.b64encode(b"admin:admin").decode()
        assert c.get("/config", headers={"Authorization": f"Basic {token}"}).status_code == 200
        wrong = base64.b64encode(b"admin:sbagliata").decode()
        assert c.get("/config", headers={"Authorization": f"Basic {wrong}"}).status_code == 401


def test_verify_credentials_roles(with_operator):
    assert verify_credentials(with_operator, "admin", "admin").is_admin
    op = verify_credentials(with_operator, "operatore", "op-pass")
    assert op.role == "operatore" and not op.is_admin
    assert verify_credentials(with_operator, "operatore", "admin") is None


def test_operator_cannot_approve_or_open_admin_pages(tool_ctx, with_operator):
    args = {"machine_id": "M01", "failure_type": "TWF", "priority": "P3", "notes": "utensile"}
    script = [tool_call("open_work_order", args), AIMessage(content="Registrato.")]
    with _app(tool_ctx, with_operator, script) as c:
        login(c, "operatore", "op-pass")
        first = c.post("/chat", json={"message": "Apri un ordine per M01"}).json()
        assert first["pending_approval"]
        assert c.post(f"/chat/{first['thread_id']}/decision", json={"approve": True}).status_code == 403
        assert c.post(f"/chat/{first['thread_id']}/decision/stream", json={"approve": True}).status_code == 403
        for url in ["/experiments/runs", "/ops/drift", "/ops/audit", "/metrics/daily"]:
            assert c.get(url).status_code == 403, url
        assert c.get("/config").json()["user"]["role"] == "operatore"
        # L'admin, nella stessa conversazione, può invece decidere.
        c.post("/auth/logout")
        login(c)
        done = c.post(f"/chat/{first['thread_id']}/decision", json={"approve": True}).json()
        assert done["answer"] == "Registrato."


def test_approval_records_who_approved(tool_ctx):
    import sqlite3

    args = {"machine_id": "M03", "failure_type": "PWF", "priority": "P2", "notes": "controllo potenza"}
    with _app(tool_ctx, script=[tool_call("open_work_order", args), AIMessage(content="Fatto.")]) as c:
        login(c)
        tid = c.post("/chat", json={"message": "Apri un ordine per M03"}).json()["thread_id"]
        c.post(f"/chat/{tid}/decision", json={"approve": True})
    with sqlite3.connect(tool_ctx.settings.db_path) as conn:
        row = conn.execute("SELECT approved_by, approved_at FROM work_orders WHERE machine_id = 'M03' "
                           "ORDER BY work_order_id DESC LIMIT 1").fetchone()
    assert row[0] == "admin" and row[1]
    audit = get_store(tool_ctx.settings.app_db_path).query(
        "SELECT action, target FROM audit_log WHERE user = 'admin' AND action = 'approve'")
    assert any(a["target"] == tid for a in audit)


def test_rate_limit(tool_ctx):
    settings = tool_ctx.settings.model_copy(update={"rate_limit_per_minute": 2})
    with _app(tool_ctx, settings, [AIMessage(content="ok")] * 5) as c:
        login(c)
        assert c.post("/chat", json={"message": "uno"}).status_code == 200
        assert c.post("/chat", json={"message": "due"}).status_code == 200
        third = c.post("/chat", json={"message": "tre"})
        assert third.status_code == 429 and "Retry-After" in third.headers


def test_daily_token_budget(tool_ctx):
    settings = tool_ctx.settings.model_copy(update={"daily_token_budget": 100})
    get_store(settings.app_db_path).record_request(user="budget-user", tokens_in=90, tokens_out=20, route="in_scope")
    budget_user = settings.model_copy(update={"operator_user": "budget-user", "operator_password": SecretStr("pw")})
    with _app(tool_ctx, budget_user) as c:
        login(c, "budget-user", "pw")
        resp = c.post("/chat", json={"message": "ciao"})
        assert resp.status_code == 429 and "Budget" in resp.json()["detail"]


def test_prod_refuses_default_password(tool_ctx):
    prod = tool_ctx.settings.model_copy(update={"env": "prod"})
    with pytest.raises(RuntimeError, match="produzione"):
        with _app(tool_ctx, prod):
            pass
    allowed = prod.model_copy(update={"allow_default_password": True})
    with _app(tool_ctx, allowed) as c:
        assert c.get("/health/live").status_code == 200


def test_security_headers(tool_ctx):
    with _app(tool_ctx) as c:
        resp = c.get("/health/live")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert "X-Request-ID" in resp.headers
        # Un X-Request-ID con caratteri strani non finisce nei log così com'è.
        weird = c.get("/health/live", headers={"X-Request-ID": "a b\nc"})
        assert weird.headers["X-Request-ID"] != "a b\nc"
