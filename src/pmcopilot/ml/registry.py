"""
registry.py — il "model registry": ogni modello addestrato resta, uno solo è in produzione.

IL PROBLEMA
    Riaddestrare sovrascrivendo artifacts/failure_model.joblib significa:
    non poter tornare indietro, non sapere quale modello ha prodotto una
    previsione, rischiare di mettere in produzione un modello peggiore.

LA SOLUZIONE (la stessa idea di MLflow Model Registry, in piccolo)
    artifacts/models/<versione>/model.joblib          ogni modello addestrato
    artifacts/models/<versione>/model_card.json
    artifacts/models/<versione>/reference_profile.json (distribuzioni di training)
    app.db, tabella model_registry                    metriche e stato:
        candidate   addestrato, non in uso
        production  quello usato da API, agente e scoring (uno solo)
        archived    era in produzione, sostituito (si può ripromuovere = rollback)

    "Promuovere" una versione copia i suoi file in artifacts/ (dove il
    predittore li legge) e aggiorna il registry. Chi promuove viene registrato
    nell'audit.

CHAMPION / CHALLENGER
    Un nuovo modello (challenger) va in produzione solo se, SUGLI STESSI DATI
    DI TEST, batte quello attuale (champion). Confrontare metriche calcolate su
    test set diversi sarebbe come confrontare due voti di esami diversi.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pmcopilot.config import Settings
from pmcopilot.store import get_store

FILES = ("model.joblib", "model_card.json", "reference_profile.json")


def version_dir(settings: Settings, version: str) -> Path:
    return settings.models_dir / version


def promote(settings: Settings, version: str, user: str) -> dict:
    """Mette in produzione una versione del registry (anche per fare rollback)."""
    src = version_dir(settings, version)
    if not (src / "model.joblib").exists():
        raise FileNotFoundError(f"Versione {version} non trovata in {settings.models_dir}")
    shutil.copy2(src / "model.joblib", settings.model_path)
    shutil.copy2(src / "model_card.json", settings.model_card_path)
    if (src / "reference_profile.json").exists():
        shutil.copy2(src / "reference_profile.json", settings.reference_profile_path)
    store = get_store(settings.app_db_path)
    store.set_production(version, user)
    store.audit(user, "promote_model", version)
    return json.loads((src / "model_card.json").read_text())


def list_models(settings: Settings) -> list[dict]:
    return get_store(settings.app_db_path).models()
