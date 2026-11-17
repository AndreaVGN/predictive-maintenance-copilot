"""
simulate.py — un "flusso sensori" finto per vedere il sistema in esercizio e il drift.

PERCHÉ
    Il dataset è una fotografia di 3 settimane. Un sistema in produzione invece
    riceve dati NUOVI ogni giorno. Questo modulo genera nuove letture per le
    macchine esistenti, con timestamp successivi all'ultima lettura, e le
    aggiunge a sensor_readings (più i relativi interventi di manutenzione).
    Così scoring, controllo del drift, alert e riaddestramento lavorano su
    dati che "arrivano" davvero.

GLI SCENARI (cosa cambia nel mondo simulato)
    normal        nessun cambiamento: il drift deve restare "ok" (controllo dei falsi allarmi)
    seasonal      arriva l'estate: aria +6 K e processo +6 K. Cambiano gli input,
                  il delta termico no -> DATA DRIFT senza impatto sui guasti
    sensor_bias   la sonda dell'aria si stara di +4 K: il valore MISURATO sale,
                  quello vero no. Il delta termico misurato scende e il modello
                  "vede" rischi HDF che non esistono -> data drift + prediction drift
    late_replacement  per carenza di ricambi gli utensili vengono sostituiti in
                  ritardo (fino a +60 minuti oltre la finestra 200-240): compaiono
                  usure mai viste nel training -> più guasti OSF/TWF, drift su
                  usura e usura x coppia
    concept       un refrigerante peggiore fa scattare l'HDF già sotto 10 K invece
                  di 8,6 K. Gli INPUT restano identici: lo vedi solo dalle prestazioni
                  (recall in calo) -> CONCEPT DRIFT

    `intensity` (0-1) scala l'effetto; con `ramp=True` l'effetto cresce
    gradualmente lungo il periodo (drift graduale invece che improvviso).
"""

from __future__ import annotations

import json
import logging
from contextlib import closing

import numpy as np
import pandas as pd

from pmcopilot import domain
from pmcopilot.config import Settings
from pmcopilot.data.database import READING_INTERVAL, _build_maintenance_log, connect
from pmcopilot.data.features import mechanical_power_w

logger = logging.getLogger(__name__)

SCENARIOS = ("normal", "seasonal", "sensor_bias", "late_replacement", "concept")


def _reference_rows(settings: Settings) -> pd.DataFrame:
    """Le letture del periodo di riferimento (quello su cui è stato addestrato il modello).

    Lo scenario "normal" deve avere LA STESSA distribuzione del training,
    altrimenti il monitoraggio segnalerebbe un drift che non esiste (falso allarme).
    Il modo più sicuro per ottenerla è ricampionare le letture storiche
    (bootstrap) invece di inventare nuove distribuzioni.
    """
    end = None
    if settings.model_card_path.exists():
        end = json.loads(settings.model_card_path.read_text()).get("train_period", [None, None])[1]
    query = "SELECT air_temp_k, process_temp_k, torque_nm, rpm FROM sensor_readings"
    params: tuple = ()
    if end:
        query += " WHERE ts <= ?"
        params = (end,)
    with closing(connect(settings.db_path)) as conn:
        return pd.read_sql_query(query, conn, params=params)


def _true_conditions(rng: np.random.Generator, base: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    """Condizioni operative VERE: righe storiche ricampionate, con un po' di rumore."""
    rows = base.iloc[rng.integers(0, len(base), n)].reset_index(drop=True)
    air = rows["air_temp_k"].to_numpy() + rng.normal(0, 0.1, n)
    process = rows["process_temp_k"].to_numpy() + rng.normal(0, 0.1, n)
    torque = np.clip(rows["torque_nm"].to_numpy() + rng.normal(0, 0.3, n), 3.8, None)
    rpm = np.clip(rows["rpm"].to_numpy() + rng.normal(0, 5, n), 1168, 2886).round()
    return {"air": air.round(1), "process": process.round(1), "torque": torque.round(1), "rpm": rpm}


def _wear_sequence(rng: np.random.Generator, types: np.ndarray, delay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Usura utensile con sostituzioni (stessa logica del generatore) e guasti TWF.

    `delay` (minuti, per riga) sposta in avanti il momento della sostituzione.
    """
    wear = np.zeros(len(types), dtype=int)
    twf = np.zeros(len(types), dtype=int)
    current = 0
    replace_at = rng.integers(domain.TOOL_WEAR_REPLACEMENT_MIN, domain.TOOL_WEAR_REPLACEMENT_MAX + 1)
    for i, t in enumerate(types):
        wear[i] = current
        current += domain.TOOL_WEAR_INCREMENT[str(t)]
        if wear[i] >= replace_at + delay[i]:
            # Oltre la fine vita l'utensile si rompe più spesso.
            p_break = 51 / 120 + (0.4 * delay[i] / 60 if delay[i] else 0.0)
            twf[i] = int(rng.random() < min(p_break, 0.95))
            current = 0
            replace_at = rng.integers(domain.TOOL_WEAR_REPLACEMENT_MIN, domain.TOOL_WEAR_REPLACEMENT_MAX + 1)
    return wear, twf


def simulate_readings(settings: Settings, hours: float = 24, scenario: str = "normal",
                      intensity: float = 1.0, ramp: bool = False, seed: int | None = None) -> dict:
    """Genera `hours` ore di nuove letture (una ogni 3 minuti) e le salva nel database."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Scenario sconosciuto '{scenario}'. Possibili: {', '.join(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    n = max(1, int(hours * 60 / 3))
    with closing(connect(settings.db_path)) as conn:
        machines = pd.read_sql_query("SELECT machine_id, machine_type FROM machines ORDER BY machine_id", conn)
        last_ts, last_id = conn.execute("SELECT MAX(ts), MAX(reading_id) FROM sensor_readings").fetchone()

    # Ogni nuova lettura va a una macchina a caso (come i turni di produzione).
    pick = rng.integers(0, len(machines), n)
    machine_ids = machines["machine_id"].to_numpy()[pick]
    types = machines["machine_type"].to_numpy()[pick]
    # Effetto dello scenario: costante, oppure crescente da 0 a `intensity` (rampa).
    effect = np.linspace(0, intensity, n) if ramp else np.full(n, intensity)

    true = _true_conditions(rng, _reference_rows(settings), n)
    delay = (60.0 * effect).round() if scenario == "late_replacement" else np.zeros(n)
    wear, twf = _wear_sequence(rng, types, delay)
    if scenario == "seasonal":
        true["air"] = (true["air"] + 6.0 * effect).round(1)
        true["process"] = (true["process"] + 6.0 * effect).round(1)

    # Etichette dai valori VERI, con le regole fisiche (eventualmente cambiate).
    hdf_limit = domain.HDF_MAX_TEMP_DIFF_K + (1.4 * effect if scenario == "concept" else 0.0)
    diff = true["process"] - true["air"]
    power = mechanical_power_w(true["torque"], true["rpm"])
    osf_limit = np.vectorize(domain.OSF_LIMIT.get)(types)
    hdf = (diff < hdf_limit) & (true["rpm"] < domain.HDF_MAX_RPM)
    pwf = (power < domain.PWF_MIN_POWER_W) | (power > domain.PWF_MAX_POWER_W)
    osf = wear * true["torque"] > osf_limit
    rnf = rng.random(n) < domain.RNF_PROBABILITY
    failure = (twf.astype(bool) | hdf | pwf | osf | rnf).astype(int)

    # Valori MISURATI: con la sonda starata, il database riceve un'aria più calda del vero.
    measured_air = true["air"] + (4.0 * effect if scenario == "sensor_bias" else 0.0)

    start = pd.Timestamp(last_ts) + READING_INTERVAL
    readings = pd.DataFrame({
        "reading_id": np.arange(int(last_id) + 1, int(last_id) + 1 + n),
        "machine_id": machine_ids,
        "product_id": [f"{t}{s}" for t, s in zip(types, rng.integers(10_000, 60_000, n), strict=True)],
        "ts": [(start + i * READING_INTERVAL).isoformat(sep=" ") for i in range(n)],
        "air_temp_k": np.round(measured_air, 1),
        "process_temp_k": true["process"],
        "rpm": true["rpm"].astype(int),
        "torque_nm": true["torque"],
        "tool_wear_min": wear,
        "machine_failure": failure,
        "twf": twf, "hdf": hdf.astype(int), "pwf": pwf.astype(int),
        "osf": osf.astype(int), "rnf": rnf.astype(int),
    })
    maintenance = _build_maintenance_log(readings, rng)
    with closing(connect(settings.db_path)) as conn:
        readings.to_sql("sensor_readings", conn, if_exists="append", index=False)
        if not maintenance.empty:
            maintenance.to_sql("maintenance_log", conn, if_exists="append", index=False)
        conn.commit()

    summary = {"scenario": scenario, "intensity": intensity, "ramp": ramp, "rows": n,
               "failures": int(failure.sum()), "from": readings["ts"].iloc[0], "to": readings["ts"].iloc[-1]}
    logger.info("Simulazione: %s", summary)
    return summary

