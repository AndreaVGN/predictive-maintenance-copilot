"""
backup.py — copie di sicurezza dei database e del modello in produzione.

PERCHÉ NON BASTA COPIARE IL FILE
    Copiare un database SQLite mentre qualcuno ci scrive può produrre una copia
    corrotta (metà prima, metà dopo una transazione). L'API di backup di SQLite
    (`conn.backup`) crea una copia CONSISTENTE anche con il database in uso.

COSA SALVIAMO
    plant.db        dati di stabilimento (sensori, interventi, ordini)
    app.db          storico applicativo (richieste, audit, esperimenti, drift)
    checkpoints.db  conversazioni e approvazioni in sospeso
    failure_model.joblib + model_card.json + reference_profile.json

    L'indice Chroma NON serve: si ricostruisce dai manuali con `pmc ingest`.

RETENTION
    Si tengono gli ultimi N backup (PMC_BACKUPS_KEEP): i più vecchi si cancellano.
    In produzione le copie vanno anche FUORI dalla macchina (storage di rete o
    cloud): un backup sullo stesso disco non protegge da un guasto del disco.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from pmcopilot.config import Settings

logger = logging.getLogger(__name__)


def create_backup(settings: Settings) -> Path:
    target = settings.backups_dir / f"backup-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    target.mkdir(parents=True, exist_ok=True)
    for db in (settings.db_path, settings.app_db_path, settings.checkpoint_db_path):
        if db.exists():
            with closing(sqlite3.connect(db)) as src, closing(sqlite3.connect(target / db.name)) as dst:
                src.backup(dst)
    for f in (settings.model_path, settings.model_card_path, settings.reference_profile_path):
        if f.exists():
            shutil.copy2(f, target / f.name)
    prune_backups(settings)
    logger.info("Backup creato in %s", target)
    return target


def prune_backups(settings: Settings) -> list[Path]:
    backups = sorted(p for p in settings.backups_dir.glob("backup-*") if p.is_dir())
    removed = backups[:-settings.backups_keep] if len(backups) > settings.backups_keep else []
    for path in removed:
        shutil.rmtree(path)
    return removed
