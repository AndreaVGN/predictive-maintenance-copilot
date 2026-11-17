"""
drift.py — accorgersi che il mondo è cambiato rispetto ai dati di addestramento.

COS'È IL DRIFT
    Un modello impara le relazioni presenti nei dati di training. In esercizio
    i dati cambiano: arriva l'estate, un sensore si stara, cambia il fornitore
    degli utensili. Tre tipi, che si misurano in modo diverso:

    data drift (covariate shift)   cambia la distribuzione degli INPUT, P(X).
        Es.: la temperatura ambiente sale di 6 K d'estate.
        Si misura confrontando le distribuzioni: PSI e test di Kolmogorov-Smirnov.
        Non sempre peggiora il modello, ma è un campanello d'allarme.

    prediction drift               cambia la distribuzione delle PREVISIONI.
        Es.: la probabilità media di guasto passa dal 4% al 15%.
        Si misura con il PSI sulle probabilità. Utile perché è sempre
        disponibile, anche quando le etichette (i guasti veri) arrivano dopo.

    concept drift                  cambia la RELAZIONE tra input e guasto, P(y|X).
        Es.: un nuovo refrigerante peggiore fa scattare l'HDF già a 9,5 K.
        Gli input possono restare identici! Lo vedi solo dalle PRESTAZIONI
        quando arrivano le etichette: recall e PR-AUC sulla finestra recente.

PSI — POPULATION STABILITY INDEX
    Si divide il range della variabile in bin usando i quantili del TRAINING
    (10 bin con ~10% dei dati ciascuno). Per ogni bin: e = % attesa (training),
    a = % attuale (finestra recente).

        PSI = Σ (a - e) · ln(a / e)

    Regola pratica del settore: < 0,1 stabile · 0,1-0,25 cambiamento moderato ·
    > 0,25 cambiamento significativo. Esempio: se nel bin "aria > 302 K" c'era il
    10% dei dati e ora c'è il 40%, quel bin da solo vale (0,4-0,1)·ln(4) ≈ 0,42.

KS — TEST DI KOLMOGOROV-SMIRNOV
    Misura la distanza massima tra le due funzioni di ripartizione (CDF).
    Restituisce una statistica D (0 = identiche, 1 = disgiunte) e un p-value.
    Con migliaia di righe anche differenze minuscole danno p-value piccoli:
    per questo decidiamo con il PSI e usiamo KS come conferma.
"""

from __future__ import annotations

import json
import logging
from contextlib import closing
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, recall_score

from pmcopilot.config import Settings
from pmcopilot.data.database import connect_readonly
from pmcopilot.data.features import (
    ENGINEERED_FEATURES,
    RAW_NUMERIC_FEATURES,
    TARGET,
    add_engineered_features,
)
from pmcopilot.ml.predictor import FailurePredictor
from pmcopilot.store import get_store

logger = logging.getLogger(__name__)

DRIFT_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
N_BINS = 10
EPS = 1e-4  # evita log(0) e divisioni per zero nei bin vuoti
SAMPLE_SIZE = 2000  # valori di training salvati per il test KS
# Per le PROBABILITÀ usiamo bin fissi su una scala sensata invece dei quantili:
# quasi tutte le probabilità sono vicine a 0, quindi i decili sarebbero bordi
# minuscoli (0,0001; 0,0002...) e un'oscillazione irrilevante sposterebbe molti
# valori di bin, gonfiando il PSI (falsi allarmi). Problema tipico delle
# distribuzioni molto asimmetriche.
PROBA_EDGES = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])


# ------------------------------------------------------------- riferimento
def build_reference_profile(train_df: pd.DataFrame, proba: np.ndarray, threshold: float,
                            seed: int = 42, test_metrics: dict | None = None) -> dict[str, Any]:
    """Fotografia delle distribuzioni di riferimento, salvata insieme al modello.

    train_df  le feature del training (riferimento per il data drift)
    proba     le probabilità del modello FINALE su dati che non ha visto (il test set):
              sono confrontabili con quelle che produrrà in esercizio.
    """
    rng = np.random.default_rng(seed)
    features = {}
    for col in DRIFT_FEATURES:
        values = train_df[col].to_numpy(dtype=float)
        # Bordi interni dei bin = quantili 10%, 20%, ..., 90% del training.
        edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        expected = _proportions(values, edges)
        sample = rng.choice(values, size=min(SAMPLE_SIZE, len(values)), replace=False)
        features[col] = {"edges": edges.tolist(), "expected": expected.tolist(),
                         "mean": float(values.mean()), "std": float(values.std()),
                         "sample": sample.round(4).tolist()}
    return {
        "features": features,
        "machine_type": train_df["machine_type"].value_counts(normalize=True).round(4).to_dict(),
        "prediction": {"edges": PROBA_EDGES.tolist(), "expected": _proportions(proba, PROBA_EDGES).tolist(),
                       "mean": float(proba.mean())},
        "positive_rate": float(train_df[TARGET].mean()),
        "threshold": threshold,
        # Prestazioni attese (sul test): il riferimento per accorgersi del concept drift.
        "test_metrics": {k: test_metrics[k] for k in ("pr_auc", "recall")} if test_metrics else {},
        "n_rows": int(len(train_df)),
    }


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Quota di valori in ogni bin. np.digitize dice in quale bin cade ogni valore."""
    counts = np.bincount(np.digitize(values, edges), minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


SEVERITY = ("ok", "warning", "alert")


def status_for(value: float, settings: Settings) -> str:
    if value >= settings.drift_psi_alert:
        return "alert"
    if value >= settings.drift_psi_warn:
        return "warning"
    return "ok"


# ------------------------------------------------------------- controllo
def load_window(settings: Settings, hours: int) -> pd.DataFrame:
    """Le letture delle ultime `hours` ore di dati (rispetto all'ultima lettura)."""
    query = """
        SELECT r.*, m.machine_type FROM sensor_readings r JOIN machines m USING (machine_id)
        WHERE r.ts >= datetime((SELECT MAX(ts) FROM sensor_readings), ?)
        ORDER BY r.ts
    """
    with closing(connect_readonly(settings.db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=(f"-{hours} hours",))
    return add_engineered_features(df)


def check_drift(settings: Settings, hours: int | None = None, save: bool = True) -> dict[str, Any]:
    """Confronta la finestra recente con il profilo di training e salva il report."""
    hours = hours or settings.drift_window_hours
    if not settings.reference_profile_path.exists():
        raise FileNotFoundError("Profilo di riferimento mancante: esegui 'pmc train'.")
    ref = json.loads(settings.reference_profile_path.read_text())
    window = load_window(settings, hours)
    if len(window) < 50:
        raise ValueError(f"Finestra troppo piccola ({len(window)} righe): servono almeno 50 letture.")

    features: dict[str, Any] = {}
    for col in DRIFT_FEATURES:
        r = ref["features"][col]
        values = window[col].to_numpy(dtype=float)
        value = psi(np.array(r["expected"]), _proportions(values, np.array(r["edges"])))
        ks = ks_2samp(np.array(r["sample"]), values)
        features[col] = {"psi": round(value, 4), "status": status_for(value, settings),
                         "ks_stat": round(float(ks.statistic), 4), "ks_pvalue": float(ks.pvalue),
                         "mean_ref": round(r["mean"], 3), "mean_now": round(float(values.mean()), 3)}

    # Prediction drift: il modello in produzione valuta la finestra.
    predictor = FailurePredictor(settings.model_path)
    proba = predictor.predict_frame(window)["failure_proba"].to_numpy()
    pr = ref["prediction"]
    pred_psi = psi(np.array(pr["expected"]), _proportions(proba, np.array(pr["edges"])))
    prediction = {"psi": round(pred_psi, 4), "status": status_for(pred_psi, settings),
                  "mean_ref": round(pr["mean"], 4), "mean_now": round(float(proba.mean()), 4),
                  "alarm_rate_now": round(float((proba >= predictor.threshold).mean()), 4)}

    # Prestazioni sulla finestra (le etichette qui sono disponibili: in un
    # impianto vero arriverebbero con ritardo, quando il guasto avviene o no).
    y = window[TARGET].to_numpy()
    performance: dict[str, Any] = {"positive_rate_ref": round(ref["positive_rate"], 4),
                                   "positive_rate_now": round(float(y.mean()), 4), "n_failures": int(y.sum())}
    perf_status = "ok"
    if 0 < y.sum() < len(y):
        performance["pr_auc"] = round(float(average_precision_score(y, proba)), 4)
        performance["recall"] = round(float(recall_score(y, proba >= predictor.threshold)), 4)
        # Calo rispetto alle prestazioni attese: la spia del CONCEPT drift.
        expected = ref.get("test_metrics", {})
        drops = [expected[m] - performance[m] for m in ("pr_auc", "recall") if m in expected]
        if drops:
            performance["max_drop"] = round(max(drops), 4)
            perf_status = "alert" if max(drops) >= 0.15 else "warning" if max(drops) >= 0.08 else "ok"
    performance["status"] = perf_status

    # Diagnosi: quale tipo di drift è presente (possono coesistere).
    feature_status = max((f["status"] for f in features.values()), key=SEVERITY.index)
    diagnosis = []
    if feature_status != "ok":
        diagnosis.append("data_drift")
    if prediction["status"] != "ok":
        diagnosis.append("prediction_drift")
    if perf_status != "ok":
        diagnosis.append("performance_drop")
    status = max((feature_status, prediction["status"], perf_status), key=SEVERITY.index)
    report = {"window_start": str(window["ts"].iloc[0]), "window_end": str(window["ts"].iloc[-1]),
              "n_rows": int(len(window)), "status": status, "features": features,
              "prediction": {**prediction, "diagnosis": diagnosis},
              "performance": performance, "model_version": predictor.version}
    if save:
        get_store(settings.app_db_path).save_drift_report(**report)
    logger.info("Controllo drift: stato %s, diagnosi %s, %d righe", status, diagnosis or "nessuna", len(window))
    return report

