"""Test della parte dati: generatore, feature numpy, database SQLite."""

import sqlite3
from contextlib import closing

import numpy as np
import pytest

from pmcopilot import domain
from pmcopilot.data.database import connect_readonly, dataset_time_range, list_machine_ids
from pmcopilot.data.features import (
    add_engineered_features,
    clean_columns,
    feature_columns,
    mechanical_power_w,
)
from pmcopilot.data.synthetic import AI4I_COLUMNS, generate_ai4i_like


def test_generator_is_deterministic_and_has_ai4i_columns():
    a = generate_ai4i_like(n=500, seed=1)
    b = generate_ai4i_like(n=500, seed=1)
    assert list(a.columns) == AI4I_COLUMNS
    assert a.equals(b)  # stesso seed -> stessi dati


def test_generator_failure_rules_are_consistent():
    df = add_engineered_features(clean_columns(generate_ai4i_like(n=5000, seed=3)))
    # Ogni HDF deve rispettare la regola fisica che l'ha generato.
    hdf = df[df["hdf"] == 1]
    assert (hdf["temp_diff_k"] < domain.HDF_MAX_TEMP_DIFF_K).all()
    assert (hdf["rpm"] < domain.HDF_MAX_RPM).all()
    # machine_failure deve essere 1 quando almeno una modalità è attiva.
    any_mode = df[["twf", "hdf", "pwf", "osf", "rnf"]].max(axis=1)
    assert (df["machine_failure"] == any_mode).all()
    # Tasso di guasto plausibile (il dataset reale è ~3.4%).
    assert 0.02 < df["machine_failure"].mean() < 0.08


def test_mechanical_power():
    # 40 Nm a 1500 rpm = 40 * 1500 * 2pi / 60 ≈ 6283 W
    assert mechanical_power_w(40, 1500) == pytest.approx(6283.19, rel=1e-4)
    arr = mechanical_power_w(np.array([10.0, 20.0]), np.array([3000, 1500]))
    assert arr.shape == (2,)
    assert arr[0] == pytest.approx(arr[1])  # stessa potenza: coppia x giri uguale


def test_feature_columns_never_include_leakage():
    cols = feature_columns(use_engineered=True)
    for leak in ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]:
        assert leak not in cols


def test_database_contents(env):
    ids = list_machine_ids(env.db_path)
    assert ids and all(i.startswith("M") for i in ids)
    first, last = dataset_time_range(env.db_path)
    assert first < last
    with closing(connect_readonly(env.db_path)) as conn:
        n_fail = conn.execute("SELECT SUM(machine_failure) FROM sensor_readings").fetchone()[0]
        n_log = conn.execute("SELECT COUNT(*) FROM maintenance_log").fetchone()[0]
        assert n_fail == n_log  # un intervento per ogni guasto
        assert conn.execute("SELECT COUNT(*) FROM v_machine_risk").fetchone()[0] == len(ids)


def test_readonly_connection_rejects_writes(env):
    with closing(connect_readonly(env.db_path)) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM machines")
