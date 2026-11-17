"""
predictor.py — il "mini modello di inferenza": carica il modello e risponde.

È usato da tre parti del progetto:
  * l'endpoint FastAPI POST /predict (valori passati dall'utente);
  * il tool dell'agente `predict_failure` (ultima lettura di una macchina,
    eventualmente con valori modificati per simulare uno scenario "what-if");
  * lo scoring batch (ml/score.py) che riempie la tabella `predictions`.

OLTRE LA PROBABILITÀ: I FATTORI DI RISCHIO
    Un 0.83 da solo non aiuta un manutentore. Aggiungiamo una spiegazione
    basata sulle soglie fisiche di domain.py (le stesse dei manuali): quale
    limite è superato o vicino a esserlo. È una forma semplice di
    explainability "di dominio", che non richiede librerie come SHAP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from pmcopilot import domain
from pmcopilot.data.features import add_engineered_features


@dataclass
class MachineReading:
    """I valori grezzi di una lavorazione (le stesse colonne del dataset)."""

    machine_type: str
    air_temp_k: float
    process_temp_k: float
    rpm: float
    torque_nm: float
    tool_wear_min: float


@dataclass
class PredictionResult:
    failure_proba: float
    predicted_failure: bool
    risk_level: str  # "basso" | "medio" | "alto"
    threshold: float
    model_version: str
    risk_factors: list[str] = field(default_factory=list)
    derived: dict[str, float] = field(default_factory=dict)  # feature fisiche calcolate

    def to_dict(self) -> dict:
        return asdict(self)


def risk_level_for(proba: float, threshold: float) -> str:
    """Tre livelli leggibili: sopra soglia = alto, oltre metà soglia = medio."""
    if proba >= threshold:
        return "alto"
    if proba >= threshold / 2:
        return "medio"
    return "basso"


def explain_risk_factors(row: pd.Series) -> list[str]:
    """Confronta i valori con le soglie fisiche e descrive cosa è fuori norma."""
    m = domain.WARNING_MARGIN
    factors: list[str] = []
    power, diff, rpm = row["power_w"], row["temp_diff_k"], row["rpm"]
    wear, strain = row["tool_wear_min"], row["wear_x_torque"]
    osf_limit = domain.OSF_LIMIT[str(row["machine_type"])]

    # PWF — potenza
    if power < domain.PWF_MIN_POWER_W or power > domain.PWF_MAX_POWER_W:
        factors.append(f"PWF: potenza {power:.0f} W fuori dalla finestra "
                       f"{domain.PWF_MIN_POWER_W:.0f}-{domain.PWF_MAX_POWER_W:.0f} W")
    elif power < domain.PWF_MIN_POWER_W * (1 + m) or power > domain.PWF_MAX_POWER_W * (1 - m):
        factors.append(f"PWF (attenzione): potenza {power:.0f} W vicina ai limiti")
    # HDF — dissipazione termica (servono entrambe le condizioni)
    if diff < domain.HDF_MAX_TEMP_DIFF_K and rpm < domain.HDF_MAX_RPM:
        factors.append(f"HDF: delta termico {diff:.1f} K < {domain.HDF_MAX_TEMP_DIFF_K} K "
                       f"con velocità {rpm:.0f} rpm < {domain.HDF_MAX_RPM} rpm")
    elif diff < domain.HDF_MAX_TEMP_DIFF_K * (1 + m) and rpm < domain.HDF_MAX_RPM * (1 + m):
        factors.append(f"HDF (attenzione): delta termico {diff:.1f} K e velocità {rpm:.0f} rpm "
                       "vicini alla zona di guasto")
    # OSF — sovraccarico
    if strain > osf_limit:
        factors.append(f"OSF: usura x coppia {strain:.0f} > limite {osf_limit:.0f} "
                       f"per macchine di tipo {row['machine_type']}")
    elif strain > osf_limit * (1 - m):
        factors.append(f"OSF (attenzione): usura x coppia {strain:.0f}, oltre il "
                       f"{1 - m:.0%} del limite {osf_limit:.0f}")
    # TWF — usura utensile
    if wear >= domain.TOOL_WEAR_REPLACEMENT_MIN:
        factors.append(f"TWF: usura utensile {wear:.0f} min nella finestra di fine vita "
                       f"({domain.TOOL_WEAR_REPLACEMENT_MIN}-{domain.TOOL_WEAR_REPLACEMENT_MAX} min)")
    elif wear >= domain.TOOL_WEAR_REPLACEMENT_MIN * (1 - m):
        factors.append(f"TWF (attenzione): usura utensile {wear:.0f} min, vicina alla sostituzione")
    return factors


class FailurePredictor:
    """Carica il bundle salvato da train.py una volta sola e lo riusa."""

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {model_path}. Esegui 'pmc train'.")
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.threshold: float = bundle["threshold"]
        self.features: list[str] = bundle["features"]
        self.version: str = bundle["version"]

    def predict_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Predizione vettoriale su molte righe (usata dallo scoring batch)."""
        df = add_engineered_features(raw)
        proba = self.pipeline.predict_proba(df[self.features])[:, 1]
        out = df.copy()
        out["failure_proba"] = proba
        out["risk_level"] = [risk_level_for(p, self.threshold) for p in proba]
        return out

    def predict_one(self, reading: MachineReading) -> PredictionResult:
        row = self.predict_frame(pd.DataFrame([asdict(reading)])).iloc[0]
        proba = float(row["failure_proba"])
        return PredictionResult(
            failure_proba=round(proba, 4),
            predicted_failure=proba >= self.threshold,
            risk_level=str(row["risk_level"]),
            threshold=round(self.threshold, 4),
            model_version=self.version,
            risk_factors=explain_risk_factors(row),
            derived={
                "temp_diff_k": round(float(row["temp_diff_k"]), 2),
                "power_w": round(float(row["power_w"]), 1),
                "wear_x_torque": round(float(row["wear_x_torque"]), 1),
            },
        )
