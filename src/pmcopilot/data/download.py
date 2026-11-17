"""
download.py — procurarsi il dataset AI4I 2020.

FLUSSO
    1. Se data/raw/ai4i2020.csv esiste già, non fa nulla (idempotente).
    2. Altrimenti prova a scaricare lo zip ufficiale da UCI e ne estrae il CSV.
    3. Se il download fallisce (offline, firewall, sito giù) genera dati
       sintetici equivalenti con data/synthetic.py.
    4. Scrive in data/raw/SOURCE.txt da dove vengono i dati, così README,
       report e documentazione possono dichiararlo in modo onesto.

    Usiamo solo la libreria standard (urllib, zipfile, io) per non aggiungere
    dipendenze per una singola richiesta HTTP.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile

import pandas as pd

from pmcopilot.config import Settings
from pmcopilot.data.synthetic import AI4I_COLUMNS, generate_ai4i_like

logger = logging.getLogger(__name__)

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)
CSV_NAME_IN_ZIP = "ai4i2020.csv"


def _download_from_uci(timeout_s: float = 30.0) -> pd.DataFrame:
    """Scarica lo zip in memoria e legge il CSV con pandas."""
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=timeout_s) as response:  # noqa: S310
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with archive.open(CSV_NAME_IN_ZIP) as csv_file:
            df = pd.read_csv(csv_file)
    missing = set(AI4I_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Il CSV scaricato non ha le colonne attese: {missing}")
    return df


def ensure_dataset(settings: Settings, force: bool = False) -> tuple[pd.DataFrame, str]:
    """Restituisce (DataFrame grezzo, sorgente) dove sorgente è 'uci' o 'synthetic'."""
    csv_path = settings.raw_csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force:
        source = (
            settings.data_source_path.read_text().strip()
            if settings.data_source_path.exists()
            else "unknown"
        )
        logger.info("Dataset già presente (%s): %s", source, csv_path)
        return pd.read_csv(csv_path), source

    source = "synthetic"
    if not settings.force_synthetic_data:
        try:
            logger.info("Scarico AI4I 2020 da UCI...")
            df = _download_from_uci()
            source = "uci"
        except Exception as exc:  # qualunque errore di rete -> piano B
            logger.warning("Download fallito (%s): genero dati sintetici equivalenti.", exc)
    if source == "synthetic":
        df = generate_ai4i_like(n=10_000, seed=settings.random_seed)

    df.to_csv(csv_path, index=False)
    settings.data_source_path.write_text(source)
    logger.info("Dataset salvato in %s (sorgente: %s, %d righe)", csv_path, source, len(df))
    return df, source
