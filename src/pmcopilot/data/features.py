"""
features.py — pulizia delle colonne (pandas) e feature engineering (numpy).

PERCHÉ ESISTE
    Il CSV originale ha nomi scomodi ("Air temperature [K]") e le variabili
    grezze non dicono direttamente al modello COSA causa un guasto.
    Qui facciamo due cose:
      1. rinominiamo le colonne in snake_case, comodo in Python e in SQL;
      2. costruiamo tre feature "fisiche" che rispecchiano le cause di guasto
         descritte in domain.py.

PERCHÉ LE FEATURE FISICHE CONTANO (punto forte da raccontare a un colloquio)
    Un albero decisionale fa fatica a imparare "coppia x giri < 3500" partendo
    da coppia e giri separati: deve approssimare una curva con tanti gradini.
    Se gli diamo direttamente la potenza, basta un taglio solo. Nella
    valutazione (ml/train.py) confrontiamo i modelli CON e SENZA queste
    feature: è un esperimento di "ablation" che dimostra il loro valore.

NUMPY IN BREVE
    Le operazioni su colonne pandas sono vettoriali: `a * b` moltiplica
    elemento per elemento tutti i 10.000 valori in una volta, in codice C,
    senza un ciclo for in Python. np.pi, np.where, np.clip lavorano allo stesso modo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmcopilot import domain

# Mappa: nome originale UCI -> nome pulito usato in tutto il progetto.
COLUMN_MAP: dict[str, str] = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "machine_type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

# Gruppi di colonne, usati dal modello e dai test.
RAW_NUMERIC_FEATURES: list[str] = [
    "air_temp_k",
    "process_temp_k",
    "rpm",
    "torque_nm",
    "tool_wear_min",
]
ENGINEERED_FEATURES: list[str] = ["temp_diff_k", "power_w", "wear_x_torque"]
CATEGORICAL_FEATURES: list[str] = ["machine_type"]
TARGET: str = "machine_failure"
# ATTENZIONE: le colonne dei singoli guasti NON devono mai entrare tra le
# feature. Sono "data leakage": se il modello vede twf=1 sa già che c'è un guasto.
FAILURE_MODE_COLUMNS: list[str] = ["twf", "hdf", "pwf", "osf", "rnf"]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rinomina le colonne e controlla che non manchi nulla."""
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {sorted(missing)}")
    out = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    # Tipi espliciti: evita sorprese (es. interi letti come float).
    int_cols = ["udi", "rpm", "tool_wear_min", TARGET, *FAILURE_MODE_COLUMNS]
    out[int_cols] = out[int_cols].astype(int)
    return out


def mechanical_power_w(torque_nm: np.ndarray | float, rpm: np.ndarray | float) -> np.ndarray:
    """Potenza meccanica P = C * omega, con omega = rpm * 2*pi / 60 (rad/s).

    Funziona sia con scalari sia con array numpy (broadcasting).
    """
    return np.asarray(torque_nm) * np.asarray(rpm) * 2.0 * np.pi / 60.0


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge le tre feature fisiche. Restituisce una COPIA (niente side effect)."""
    out = df.copy()
    # Delta termico: se è piccolo il calore non si smaltisce (guasto HDF).
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    # Potenza: fuori dalla finestra 3500-9000 W -> guasto PWF.
    out["power_w"] = mechanical_power_w(out["torque_nm"].to_numpy(), out["rpm"].to_numpy())
    # Sforzo cumulato: usura x coppia oltre soglia -> guasto OSF.
    out["wear_x_torque"] = out["tool_wear_min"] * out["torque_nm"]
    return out


def feature_columns(use_engineered: bool = True) -> list[str]:
    """Elenco delle colonne in ingresso al modello."""
    numeric = RAW_NUMERIC_FEATURES + (ENGINEERED_FEATURES if use_engineered else [])
    return numeric + CATEGORICAL_FEATURES


def osf_limit_for(machine_type: str) -> float:
    """Soglia di sovraccarico per tipo di macchina (vedi domain.OSF_LIMIT)."""
    return domain.OSF_LIMIT[machine_type]
