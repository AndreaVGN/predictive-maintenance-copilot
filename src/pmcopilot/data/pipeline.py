"""
pipeline.py — orchestrazione della parte dati: CSV -> pulizia -> database.

È il primo passo di `pmc setup`. Ogni funzione chiamata qui è testabile da
sola; questo file si limita a metterle in fila e a registrare cosa succede.
"""

from __future__ import annotations

import logging

from pmcopilot.config import Settings
from pmcopilot.data.database import build_database
from pmcopilot.data.download import ensure_dataset
from pmcopilot.data.features import clean_columns

logger = logging.getLogger(__name__)


def prepare_data(settings: Settings, force_download: bool = False) -> dict[str, int | str]:
    raw_df, source = ensure_dataset(settings, force=force_download)
    clean_df = clean_columns(raw_df)
    counts = build_database(clean_df, settings.db_path, seed=settings.random_seed)
    logger.info("Database creato in %s: %s", settings.db_path, counts)
    return {"source": source, **counts}
