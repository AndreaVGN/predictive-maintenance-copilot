"""
conftest.py — "fixture" condivise da tutti i test (pytest le trova da solo).

Una fixture prepara qualcosa che serve ai test e lo passa come argomento:
    def test_x(env):   <- pytest vede "env" e chiama la fixture env()

scope="session" significa: preparala UNA volta per tutta la sessione di test.
Costruire database, modello e indice richiede qualche secondo, quindi lo
facciamo una volta sola, in una cartella temporanea che non sporca il progetto.
Niente Ollama: embedding finti e LLM finto (vedi fakes.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))  # per "from fakes import ..."

from pmcopilot.agent.llm import get_embeddings  # noqa: E402
from pmcopilot.agent.tools import ToolContext  # noqa: E402
from pmcopilot.config import Settings  # noqa: E402
from pmcopilot.data.database import build_database, list_machine_ids  # noqa: E402
from pmcopilot.data.features import clean_columns  # noqa: E402
from pmcopilot.data.synthetic import generate_ai4i_like  # noqa: E402
from pmcopilot.ml.predictor import FailurePredictor  # noqa: E402
from pmcopilot.ml.score import score_recent_readings  # noqa: E402
from pmcopilot.ml.train import train_and_evaluate  # noqa: E402
from pmcopilot.rag.ingest import build_vector_store  # noqa: E402
from pmcopilot.rag.retriever import ManualRetriever  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    base = tmp_path_factory.mktemp("pmc")
    s = Settings(
        data_dir=base / "data",
        artifacts_dir=base / "artifacts",
        reports_dir=base / "reports",
        logs_dir=base / "logs",
        eval_dir=ROOT / "eval",
        manuals_dir=ROOT / "data" / "manuals",
        backups_dir=base / "backups",
        embedding_provider="fake",
        force_synthetic_data=True,
        router_enabled=False,
        langfuse_enabled=False,
        checkpointer="memory",
        scheduler_enabled=False,
        rate_limit_per_minute=1000,
        log_to_file=False,
    )
    s.ensure_dirs()
    return s


@pytest.fixture(scope="session")
def env(settings: Settings) -> Settings:
    """Ambiente completo: database (dataset ridotto), modello, previsioni, indice."""
    df = clean_columns(generate_ai4i_like(n=3000, seed=7))
    build_database(df, settings.db_path, seed=7)
    train_and_evaluate(settings, quick=True)
    score_recent_readings(settings, last_n_per_machine=10)
    build_vector_store(settings, get_embeddings(settings), full=True)
    return settings


def login(client, username: str = "admin", password: str = "admin") -> None:
    """Autentica un TestClient: il cookie di sessione resta nel client."""
    resp = client.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text


@pytest.fixture(scope="session")
def tool_ctx(env: Settings) -> ToolContext:
    return ToolContext(
        settings=env,
        predictor=FailurePredictor(env.model_path),
        retriever=ManualRetriever(env, get_embeddings(env)),
        machine_ids=set(list_machine_ids(env.db_path)),
    )
