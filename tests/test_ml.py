"""Test del modello: artefatti salvati, predizioni sensate, fattori di rischio."""

import json

import pandas as pd

from pmcopilot.data.features import add_engineered_features
from pmcopilot.ml.predictor import (
    FailurePredictor,
    MachineReading,
    explain_risk_factors,
    risk_level_for,
)


def test_training_outputs_exist(env):
    assert env.model_path.exists()
    card = json.loads(env.model_card_path.read_text())
    assert 0 < card["threshold"] < 1
    assert card["test_metrics"]["pr_auc"] > 0.3  # molto sopra il caso (~0.05)
    assert (env.figures_dir / "model_pr_curve.png").exists()
    assert (env.reports_dir / "model_report.md").exists()


def test_prediction_risky_vs_safe(env):
    predictor = FailurePredictor(env.model_path)
    safe = predictor.predict_one(MachineReading("M", 300.0, 310.0, 1500, 40.0, 20))
    risky = predictor.predict_one(MachineReading("L", 302.0, 310.5, 1300, 65.0, 215))
    assert 0 <= safe.failure_proba <= 1
    assert risky.failure_proba > safe.failure_proba
    assert risky.risk_factors  # deve spiegare perché


def test_risk_level_bands():
    assert risk_level_for(0.9, 0.5) == "alto"
    assert risk_level_for(0.3, 0.5) == "medio"
    assert risk_level_for(0.1, 0.5) == "basso"


def test_explain_risk_factors_power():
    row = add_engineered_features(pd.DataFrame([{
        "machine_type": "M", "air_temp_k": 300, "process_temp_k": 310,
        "rpm": 2000, "torque_nm": 15, "tool_wear_min": 10}])).iloc[0]
    factors = explain_risk_factors(row)
    assert any(f.startswith("PWF") for f in factors)  # 15 Nm x 2000 rpm ≈ 3142 W < 3500
