"""
database.py — il "gestionale di stabilimento" in SQLite.

PERCHÉ ESISTE
    In un'azienda vera i dati dei sensori non stanno in un CSV: stanno in un
    database, accanto all'anagrafica macchine e al registro manutenzioni.
    Qui simuliamo questo scenario:
      * il dataset AI4I diventa la tabella `sensor_readings`;
      * aggiungiamo uno strato realistico: 20 macchine su 3 linee, timestamp,
        un registro interventi e una tabella di ordini di lavoro.
    L'agente interrogherà queste tabelle con SQL (in SOLA LETTURA).

PERCHÉ SQLITE
    È un database completo dentro un singolo file, incluso nella libreria
    standard di Python: zero installazioni. Le query sono SQL standard, quindi
    passare a PostgreSQL in futuro richiede poche modifiche.

SCHEMA (relazioni)
    machines 1 --- N sensor_readings
    machines 1 --- N maintenance_log      (ogni guasto genera un intervento)
    machines 1 --- N predictions          (scritte dal modello ML, ml/score.py)
    machines 1 --- N work_orders          (scritti dall'agente, con approvazione umana)
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pmcopilot import domain

# Le letture partono da questa data, una lavorazione ogni 3 minuti:
# 10.000 righe = circa 21 giorni di produzione.
READINGS_START = datetime(2026, 10, 1, 6, 0, 0)
READING_INTERVAL = timedelta(minutes=3)
READINGS_PER_MACHINE = 500  # usato per decidere quante macchine creare per tipo

LINE_BY_TYPE = {"L": "Linea 1", "M": "Linea 2", "H": "Linea 3"}
TECHNICIANS = ["Rossi", "Bianchi", "Ferrari", "Galli", "Conti"]

# Azione tipica e range di fermo macchina (ore) per ogni modalità di guasto.
MAINTENANCE_PROFILE: dict[str, tuple[str, float, float]] = {
    "TWF": ("Sostituzione utensile", 0.5, 2.0),
    "HDF": ("Pulizia scambiatore e verifica ventola di raffreddamento", 2.0, 6.0),
    "PWF": ("Diagnosi azionamento e taratura parametri di taglio", 3.0, 8.0),
    "OSF": ("Sostituzione utensile e riduzione carico di taglio", 2.0, 5.0),
    "RNF": ("Ispezione generale secondo checklist", 1.0, 10.0),
    "UNK": ("Ispezione generale secondo checklist", 1.0, 4.0),
}

SCHEMA_SQL = """
-- Anagrafica delle macchine
CREATE TABLE machines (
    machine_id    TEXT PRIMARY KEY,               -- es. 'M01'
    machine_type  TEXT NOT NULL CHECK (machine_type IN ('L', 'M', 'H')),
    line          TEXT NOT NULL,                  -- linea produttiva
    installed_on  DATE NOT NULL
);

-- Una riga per lavorazione: i dati del dataset AI4I 2020
CREATE TABLE sensor_readings (
    reading_id       INTEGER PRIMARY KEY,         -- = UDI del dataset
    machine_id       TEXT NOT NULL REFERENCES machines(machine_id),
    product_id       TEXT NOT NULL,
    ts               TIMESTAMP NOT NULL,
    air_temp_k       REAL NOT NULL,
    process_temp_k   REAL NOT NULL,
    rpm              INTEGER NOT NULL,
    torque_nm        REAL NOT NULL,
    tool_wear_min    INTEGER NOT NULL,
    machine_failure  INTEGER NOT NULL CHECK (machine_failure IN (0, 1)),
    twf INTEGER NOT NULL, hdf INTEGER NOT NULL, pwf INTEGER NOT NULL,
    osf INTEGER NOT NULL, rnf INTEGER NOT NULL
);
CREATE INDEX idx_readings_machine_ts ON sensor_readings(machine_id, ts);

-- Registro degli interventi di manutenzione dopo un guasto
CREATE TABLE maintenance_log (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id    TEXT NOT NULL REFERENCES machines(machine_id),
    reading_id    INTEGER REFERENCES sensor_readings(reading_id),
    ts            TIMESTAMP NOT NULL,
    failure_type  TEXT NOT NULL,                  -- TWF, HDF, PWF, OSF, RNF, UNK
    action        TEXT NOT NULL,
    downtime_h    REAL NOT NULL,
    technician    TEXT NOT NULL
);
CREATE INDEX idx_maint_machine ON maintenance_log(machine_id, ts);

-- Previsioni del modello ML (scritte da ml/score.py)
CREATE TABLE predictions (
    prediction_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id     TEXT NOT NULL REFERENCES machines(machine_id),
    reading_id     INTEGER NOT NULL REFERENCES sensor_readings(reading_id),
    ts             TIMESTAMP NOT NULL,
    failure_proba  REAL NOT NULL,
    risk_level     TEXT NOT NULL CHECK (risk_level IN ('basso', 'medio', 'alto')),
    model_version  TEXT NOT NULL,
    scored_at      TIMESTAMP NOT NULL
);
CREATE INDEX idx_pred_machine ON predictions(machine_id, ts);

-- Ordini di lavoro aperti dall'agente (solo dopo approvazione umana)
CREATE TABLE work_orders (
    work_order_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id     TEXT NOT NULL REFERENCES machines(machine_id),
    created_at     TIMESTAMP NOT NULL,
    failure_type   TEXT NOT NULL,
    priority       TEXT NOT NULL CHECK (priority IN ('P1', 'P2', 'P3')),
    notes          TEXT,
    status         TEXT NOT NULL DEFAULT 'aperto',
    approved_by    TEXT,                          -- chi ha approvato (audit)
    approved_at    TIMESTAMP
);

-- VISTE: query pronte che semplificano la vita all'agente (e a te)
-- Ultima lettura di ogni macchina
CREATE VIEW v_latest_reading AS
SELECT r.*
FROM sensor_readings r
JOIN (SELECT machine_id, MAX(ts) AS max_ts FROM sensor_readings GROUP BY machine_id) last
  ON r.machine_id = last.machine_id AND r.ts = last.max_ts;

-- Conteggio guasti per macchina e modalità
CREATE VIEW v_failures_by_machine AS
SELECT m.machine_id, m.machine_type, m.line,
       COUNT(r.reading_id)            AS n_readings,
       COALESCE(SUM(r.machine_failure), 0) AS n_failures,
       COALESCE(SUM(r.twf), 0) AS twf, COALESCE(SUM(r.hdf), 0) AS hdf,
       COALESCE(SUM(r.pwf), 0) AS pwf, COALESCE(SUM(r.osf), 0) AS osf,
       COALESCE(SUM(r.rnf), 0) AS rnf
FROM machines m LEFT JOIN sensor_readings r ON r.machine_id = m.machine_id
GROUP BY m.machine_id, m.machine_type, m.line;

-- Rischio più recente stimato dal modello per ogni macchina
CREATE VIEW v_machine_risk AS
SELECT p.machine_id, m.machine_type, m.line, p.ts, p.failure_proba, p.risk_level, p.model_version
FROM predictions p
JOIN machines m ON m.machine_id = p.machine_id
JOIN (SELECT machine_id, MAX(ts) AS max_ts FROM predictions GROUP BY machine_id) last
  ON p.machine_id = last.machine_id AND p.ts = last.max_ts;
"""

# Tabelle e viste che l'agente può leggere (usata dal guardrail SQL).
READABLE_TABLES: frozenset[str] = frozenset(
    {
        "machines",
        "sensor_readings",
        "maintenance_log",
        "predictions",
        "work_orders",
        "v_latest_reading",
        "v_failures_by_machine",
        "v_machine_risk",
    }
)


# ----------------------------------------------------------------- connessioni
# Migrazioni: modifiche allo schema di un database GIÀ esistente. In produzione
# non si ricrea il database a ogni rilascio: si applicano solo le differenze.
# Ogni migrazione è idempotente (controlla prima se serve). Strumenti dedicati
# come Alembic fanno lo stesso con uno storico numerato delle versioni.
MIGRATIONS: list[tuple[str, str, str]] = [
    # (tabella, colonna, definizione)
    ("work_orders", "approved_by", "TEXT"),
    ("work_orders", "approved_at", "TIMESTAMP"),
]


def migrate_plant_db(db_path: Path) -> list[str]:
    """Aggiunge le colonne mancanti a un database creato con una versione precedente."""
    applied = []
    if not db_path.exists():
        return applied
    with closing(sqlite3.connect(db_path)) as conn:
        for table, column, definition in MIGRATIONS:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if existing and column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                applied.append(f"{table}.{column}")
        conn.commit()
    return applied


def connect(db_path: Path) -> sqlite3.Connection:
    """Connessione in lettura/scrittura: la usano SOLO il codice di build e i tool fidati."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Connessione in SOLA LETTURA: è quella data all'agente per il text-to-SQL.

    Due livelli di protezione, indipendenti dal guardrail che analizza la query:
      * mode=ro nell'URI: SQLite apre il file in sola lettura a livello di file;
      * PRAGMA query_only: rifiuta qualunque scrittura anche se arrivasse.
    È il principio della "difesa in profondità": se un livello fallisce,
    l'altro tiene.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database non trovato: {db_path}. Esegui prima 'pmc setup'.")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    return conn


# ---------------------------------------------------------------------- build
def _assign_machines(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crea l'anagrafica macchine e assegna ogni lettura a una macchina del suo tipo.

    Con 10.000 righe (50% L, 30% M, 20% H) ottieni ~10 macchine L, 6 M, 4 H.
    Le letture di un tipo vengono distribuite a turno (round robin) tra le
    macchine di quel tipo, mantenendo l'ordine temporale.
    """
    machines_rows = []
    machine_ids = np.empty(len(df), dtype=object)
    next_number = 1
    for machine_type in domain.MACHINE_TYPES:
        idx = np.flatnonzero(df["machine_type"].to_numpy() == machine_type)
        if len(idx) == 0:
            continue
        n_machines = max(1, round(len(idx) / READINGS_PER_MACHINE))
        ids = [f"M{next_number + k:02d}" for k in range(n_machines)]
        next_number += n_machines
        machine_ids[idx] = [ids[k % n_machines] for k in range(len(idx))]
        for machine_id in ids:
            installed = datetime(2019, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 6)))
            machines_rows.append(
                {
                    "machine_id": machine_id,
                    "machine_type": machine_type,
                    "line": LINE_BY_TYPE[machine_type],
                    "installed_on": installed.date().isoformat(),
                }
            )
    readings = df.copy()
    readings["machine_id"] = machine_ids
    return pd.DataFrame(machines_rows), readings


def _build_maintenance_log(readings: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per ogni guasto crea un intervento, qualche ora dopo, con un tecnico e un fermo."""
    failures = readings[readings["machine_failure"] == 1]
    rows = []
    for _, r in failures.iterrows():
        modes = [m.upper() for m in ("twf", "hdf", "pwf", "osf", "rnf") if r[m] == 1]
        failure_type = modes[0] if modes else "UNK"
        action, low, high = MAINTENANCE_PROFILE[failure_type]
        ts = pd.Timestamp(r["ts"]) + pd.Timedelta(hours=float(rng.uniform(0.5, 6.0)))
        rows.append(
            {
                "machine_id": r["machine_id"],
                "reading_id": int(r["reading_id"]),
                "ts": ts.isoformat(sep=" ", timespec="seconds"),
                "failure_type": failure_type,
                "action": action,
                "downtime_h": round(float(rng.uniform(low, high)), 1),
                "technician": str(rng.choice(TECHNICIANS)),
            }
        )
    return pd.DataFrame(rows)


def build_database(clean_df: pd.DataFrame, db_path: Path, seed: int = 42) -> dict[str, int]:
    """Ricostruisce da zero il database a partire dal DataFrame pulito.

    Restituisce il numero di righe per tabella (utile per log e test).
    """
    rng = np.random.default_rng(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # ricostruzione completa: idempotente e senza stati strani

    machines, readings = _assign_machines(clean_df, rng)
    # Timestamp: una lavorazione ogni 3 minuti, nell'ordine originale del dataset.
    readings["ts"] = [
        (READINGS_START + i * READING_INTERVAL).isoformat(sep=" ") for i in range(len(readings))
    ]
    readings = readings.rename(columns={"udi": "reading_id"})
    reading_cols = [
        "reading_id", "machine_id", "product_id", "ts", "air_temp_k", "process_temp_k",
        "rpm", "torque_nm", "tool_wear_min", "machine_failure",
        "twf", "hdf", "pwf", "osf", "rnf",
    ]  # fmt: skip
    readings = readings[reading_cols]
    maintenance = _build_maintenance_log(readings, rng)

    with closing(connect(db_path)) as conn:
        conn.executescript(SCHEMA_SQL)
        # pandas.to_sql inserisce un DataFrame in una tabella esistente.
        # if_exists="append" perché lo schema l'abbiamo già creato noi (con i vincoli).
        machines.to_sql("machines", conn, if_exists="append", index=False)
        readings.to_sql("sensor_readings", conn, if_exists="append", index=False)
        if not maintenance.empty:
            maintenance.to_sql("maintenance_log", conn, if_exists="append", index=False)
        conn.commit()

    return {
        "machines": len(machines),
        "sensor_readings": len(readings),
        "maintenance_log": len(maintenance),
    }


# ------------------------------------------------------------- helper di lettura
def load_readings(db_path: Path) -> pd.DataFrame:
    """Legge le letture con il tipo macchina (JOIN) in un DataFrame: SQL -> pandas."""
    query = """
        SELECT r.*, m.machine_type, m.line
        FROM sensor_readings r
        JOIN machines m ON m.machine_id = r.machine_id
        ORDER BY r.ts
    """
    with closing(connect_readonly(db_path)) as conn:
        return pd.read_sql_query(query, conn, parse_dates=["ts"])


def list_machine_ids(db_path: Path) -> list[str]:
    with closing(connect_readonly(db_path)) as conn:
        return [row[0] for row in conn.execute("SELECT machine_id FROM machines ORDER BY machine_id")]


def dataset_time_range(db_path: Path) -> tuple[str, str]:
    """Prima e ultima lettura: l'agente usa l'ultima come 'adesso'."""
    with closing(connect_readonly(db_path)) as conn:
        first, last = conn.execute("SELECT MIN(ts), MAX(ts) FROM sensor_readings").fetchone()
    return first, last


def describe_schema(db_path: Path) -> str:
    """Descrizione testuale dello schema, da inserire nel prompt del tool SQL.

    Legge le colonne direttamente dal database (PRAGMA table_info), così la
    descrizione è sempre allineata allo schema reale.
    """
    lines = []
    with closing(connect_readonly(db_path)) as conn:
        for name in sorted(READABLE_TABLES):
            cols = conn.execute(f"PRAGMA table_info({name})").fetchall()  # nome fidato, non input utente
            col_list = ", ".join(f"{c[1]} {c[2]}".strip() for c in cols)
            kind = "vista" if name.startswith("v_") else "tabella"
            lines.append(f"- {name} ({kind}): {col_list}")
    return "\n".join(lines)
