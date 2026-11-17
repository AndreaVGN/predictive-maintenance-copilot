"""
Test del monitoraggio del modello: PSI, simulazione e diagnosi del drift,
registry dei modelli (promozione e rollback).

La simulazione AGGIUNGE righe al database: per non sporcare l'ambiente
condiviso dagli altri test, qui si costruisce un ambiente separato (ml_env).
Ogni scenario aggiunge 24 ore e il controllo guarda le ultime 24: le finestre
non si sovrappongono, quindi l'ordine dei test non conta.
"""

import json

import numpy as np
import pytest

from pmcopilot.config import Settings
from pmcopilot.data.database import build_database
from pmcopilot.data.features import clean_columns
from pmcopilot.data.synthetic import generate_ai4i_like
from pmcopilot.ml.drift import check_drift, psi
from pmcopilot.ml.registry import promote
from pmcopilot.ml.score import score_recent_readings
from pmcopilot.ml.simulate import simulate_readings
from pmcopilot.ml.train import train_and_evaluate
from pmcopilot.store import get_store


@pytest.fixture(scope="module")
def ml_env(settings: Settings, tmp_path_factory) -> Settings:
    base = tmp_path_factory.mktemp("drift")
    s = settings.model_copy(update={"data_dir": base / "data", "artifacts_dir": base / "artifacts",
                                    "reports_dir": base / "reports", "backups_dir": base / "backups"})
    s.ensure_dirs()
    build_database(clean_columns(generate_ai4i_like(n=4000, seed=11)), s.db_path, seed=11)
    train_and_evaluate(s, quick=True)
    return s


# -------------------------------------------------------------------- PSI
def test_psi_is_zero_for_identical_and_grows_with_shift():
    p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    assert psi(p, p) == pytest.approx(0.0)
    small = psi(p, np.array([0.12, 0.2, 0.38, 0.2, 0.1]))
    large = psi(p, np.array([0.02, 0.08, 0.3, 0.3, 0.3]))
    assert 0 < small < 0.1 < 0.25 < large
    # Un bin vuoto non deve produrre infinito (log di zero).
    assert np.isfinite(psi(p, np.array([0.0, 0.3, 0.4, 0.2, 0.1])))


def test_reference_profile_saved_with_model(ml_env):
    profile = json.loads(ml_env.reference_profile_path.read_text())
    assert {"features", "prediction", "test_metrics", "threshold"} <= profile.keys()
    assert "air_temp_k" in profile["features"] and profile["features"]["air_temp_k"]["edges"]


# ------------------------------------------------------- scenari di drift
def _run(settings: Settings, scenario: str, intensity: float = 1.0) -> dict:
    simulate_readings(settings, hours=24, scenario=scenario, intensity=intensity, seed=3)
    score_recent_readings(settings)
    return check_drift(settings, hours=24)


def test_normal_scenario_raises_no_false_alarm(ml_env):
    report = _run(ml_env, "normal")
    assert report["status"] == "ok", report
    assert "data_drift" not in report["prediction"]["diagnosis"]


def test_seasonal_scenario_is_data_drift_on_temperatures(ml_env):
    report = _run(ml_env, "seasonal")
    assert "data_drift" in report["prediction"]["diagnosis"]
    assert report["features"]["air_temp_k"]["status"] == "alert"
    assert report["features"]["rpm"]["status"] == "ok"  # gli altri sensori non cambiano


def test_late_replacement_shows_up_on_tool_wear(ml_env):
    report = _run(ml_env, "late_replacement", intensity=2.0)
    assert report["features"]["tool_wear_min"]["psi"] > report["features"]["rpm"]["psi"]
    assert report["status"] != "ok"


def test_concept_drift_is_seen_only_through_performance(ml_env):
    report = _run(ml_env, "concept", intensity=1.5)
    assert "performance_drop" in report["prediction"]["diagnosis"]
    assert "data_drift" not in report["prediction"]["diagnosis"]  # i sensori sembrano normali


def test_drift_reports_are_kept(ml_env):
    reports = get_store(ml_env.app_db_path).drift_reports()
    assert len(reports) >= 4
    assert reports[0]["features"] and "psi" in reports[0]["prediction"]


def test_unknown_scenario_is_rejected(ml_env):
    with pytest.raises(ValueError, match="Scenario sconosciuto"):
        simulate_readings(ml_env, scenario="meteorite")


# ----------------------------------------------------------- registry
def test_registry_champion_challenger_and_rollback(ml_env):
    store = get_store(ml_env.app_db_path)
    first = store.production_model()
    assert first is not None

    challenger = train_and_evaluate(ml_env, quick=True, promote_policy="never")
    assert not challenger.promoted
    assert store.production_model()["version"] == first["version"]  # il campione resta in produzione
    assert challenger.champion_metrics is not None  # confrontato sullo stesso test set

    promote(ml_env, challenger.model_version, user="test")
    assert store.production_model()["version"] == challenger.model_version
    card = json.loads(ml_env.model_card_path.read_text())
    assert card["model_version"] == challenger.model_version

    promote(ml_env, first["version"], user="test")  # rollback
    assert store.production_model()["version"] == first["version"]
    statuses = {m["version"]: m["status"] for m in store.models()}
    assert statuses[challenger.model_version] == "archived"
    with pytest.raises(FileNotFoundError):
        promote(ml_env, "versione-inesistente", user="test")
