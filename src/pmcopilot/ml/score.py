"""
score.py — scoring batch: il modello valuta le letture recenti di ogni macchina
e scrive i risultati nella tabella `predictions`.

In un impianto vero questo script girerebbe a intervalli regolari (es. ogni
ora, con cron o un orchestratore). L'agente poi legge la vista v_machine_risk
invece di ricalcolare tutto a ogni domanda: è più veloce e lascia uno storico.
"""

from __future__ import annotations

import logging
from contextlib import closing
from datetime import UTC, datetime

import pandas as pd

from pmcopilot.config import Settings
from pmcopilot.data.database import connect
from pmcopilot.ml.predictor import FailurePredictor

logger = logging.getLogger(__name__)


def score_recent_readings(settings: Settings, last_n_per_machine: int = 48) -> int:
    """Valuta le ultime `last_n_per_machine` letture di ogni macchina."""
    predictor = FailurePredictor(settings.model_path)
    # Window function SQL: ROW_NUMBER() numera le letture di ogni macchina
    # dalla più recente; teniamo le prime N. Utile da saper scrivere a mano!
    query = """
        SELECT * FROM (
            SELECT r.reading_id, r.machine_id, r.ts, m.machine_type,
                   r.air_temp_k, r.process_temp_k, r.rpm, r.torque_nm, r.tool_wear_min,
                   ROW_NUMBER() OVER (PARTITION BY r.machine_id ORDER BY r.ts DESC) AS rn
            FROM sensor_readings r JOIN machines m ON m.machine_id = r.machine_id
        ) WHERE rn <= ?
    """
    with closing(connect(settings.db_path)) as conn:
        recent = pd.read_sql_query(query, conn, params=(last_n_per_machine,))
        scored = predictor.predict_frame(recent)
        out = pd.DataFrame(
            {
                "machine_id": scored["machine_id"],
                "reading_id": scored["reading_id"],
                "ts": scored["ts"],
                "failure_proba": scored["failure_proba"].round(4),
                "risk_level": scored["risk_level"],
                "model_version": predictor.version,
                "scored_at": datetime.now(UTC).isoformat(sep=" ", timespec="seconds"),
            }
        )
        conn.execute("DELETE FROM predictions")  # rigeneriamo lo scoring da zero
        out.to_sql("predictions", conn, if_exists="append", index=False)
        conn.commit()
    logger.info("Scoring completato: %d predizioni (%s)", len(out), predictor.version)
    return len(out)
