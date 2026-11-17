"""
store.py — il database dell'APPLICAZIONE (app.db): tutto ciò che deve sopravvivere
a un riavvio e restare come storico.

DUE DATABASE, DUE RUOLI
    plant.db  dati di stabilimento: sensori, guasti, interventi. L'agente li
              legge con SQL (in sola lettura).
    app.db    dati di esercizio del sistema: richieste, costi, audit delle
              approvazioni, feedback, valutazioni, drift, alert, registry dei
              modelli. L'agente NON lo vede: separare i file è il modo più
              semplice per garantire che una query generata dall'LLM non possa
              leggere chi ha approvato cosa, né i risultati delle valutazioni.

TABELLE
    requests        una riga per domanda: latenza, token, costo, tool, flag
    audit_log       chi ha fatto cosa (login, approvazioni, promozioni modello)
    feedback        pollice su/giù degli utenti sulle risposte
    eval_runs       ogni esecuzione di una valutazione, con la configurazione
    eval_cases      i risultati caso per caso di ogni esecuzione
    drift_reports   i controlli periodici del drift
    alerts          gli allarmi generati dalle regole di monitoraggio
    model_registry  le versioni del modello ML e quale è in produzione

PERCHÉ SQLITE (E QUANDO NO)
    Un file, zero installazioni, transazioni ACID. La modalità WAL (Write-Ahead
    Logging) permette letture mentre qualcuno scrive. Con più istanze dell'API
    su macchine diverse servirebbe PostgreSQL: le query qui sono SQL standard,
    quindi il passaggio riguarda solo questo file (vedi la guida, capitolo Produzione).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, request_id TEXT, user TEXT, thread_id TEXT, kind TEXT,
    route TEXT, latency_ms REAL, llm_calls INTEGER, tokens_in INTEGER, tokens_out INTEGER,
    context_peak INTEGER, cost_usd REAL, tools TEXT, guard_flags TEXT,
    error TEXT, llm TEXT, prompt_version TEXT, trace_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_user_ts ON requests(user, ts);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, user TEXT NOT NULL, action TEXT NOT NULL, target TEXT, details TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, user TEXT, request_id TEXT, thread_id TEXT,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)), comment TEXT
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, label TEXT,
    config TEXT, prompt_version TEXT, prompt_text TEXT, summary TEXT, user TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_kind_ts ON eval_runs(kind, ts);

CREATE TABLE IF NOT EXISTS eval_cases (
    run_id TEXT NOT NULL REFERENCES eval_runs(id), case_id TEXT NOT NULL,
    question TEXT, data TEXT, PRIMARY KEY (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, window_start TEXT, window_end TEXT, n_rows INTEGER,
    status TEXT, features TEXT, prediction TEXT, performance TEXT, model_version TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, rule TEXT NOT NULL, severity TEXT NOT NULL, message TEXT,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
    version TEXT PRIMARY KEY, ts TEXT NOT NULL, algorithm TEXT, feature_set TEXT,
    split TEXT, metrics TEXT, data_source TEXT, path TEXT,
    status TEXT NOT NULL DEFAULT 'candidate', promoted_at TEXT, promoted_by TEXT
);
"""

JSON_COLUMNS = {"tools", "guard_flags", "details", "config", "summary", "data", "features",
                "prediction", "performance", "metrics"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for key in JSON_COLUMNS & out.keys():
        if isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    return out


class AppStore:
    """Accesso thread-safe ad app.db. Una connessione per operazione, protetta da lock."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _insert(self, table: str, row: dict[str, Any]) -> int:
        data = {k: (json.dumps(v, ensure_ascii=False, default=str) if k in JSON_COLUMNS else v)
                for k, v in row.items()}
        cols = ", ".join(data)
        marks = ", ".join("?" for _ in data)
        with self._conn() as conn:
            cur = conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(data.values()))
            return int(cur.lastrowid or 0)

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._conn() as conn:
            return [_decode(r) for r in conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------- richieste
    def record_request(self, **row: Any) -> int:
        row.setdefault("ts", now_iso())
        return self._insert("requests", row)

    def tokens_used_today(self, user: str) -> int:
        start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")
        rows = self.query("SELECT COALESCE(SUM(tokens_in + tokens_out), 0) AS n FROM requests "
                          "WHERE user = ? AND ts >= ?", (user, start))
        return int(rows[0]["n"])

    def request_stats(self, minutes: int | None = None) -> dict[str, Any]:
        """Statistiche aggregate su tutte le richieste (o sugli ultimi N minuti)."""
        where, params = "", ()
        if minutes:
            where = "WHERE ts >= ?"
            params = ((datetime.now(UTC) - timedelta(minutes=minutes)).isoformat(timespec="seconds"),)
        rows = self.query(f"SELECT * FROM requests {where} ORDER BY ts", params)
        return {"rows": rows}

    # ----------------------------------------------------------------- audit
    def audit(self, user: str, action: str, target: str | None = None, **details: Any) -> None:
        self._insert("audit_log", {"ts": now_iso(), "user": user, "action": action,
                                   "target": target, "details": details or None})

    # -------------------------------------------------------------- feedback
    def add_feedback(self, user: str, request_id: str, thread_id: str | None, rating: int,
                     comment: str | None) -> int:
        return self._insert("feedback", {"ts": now_iso(), "user": user, "request_id": request_id,
                                         "thread_id": thread_id, "rating": rating, "comment": comment})

    # ----------------------------------------------------------- valutazioni
    def save_eval_run(self, kind: str, summary: dict, cases: list[dict], *, config: dict,
                      label: str | None = None, prompt_version: str | None = None,
                      prompt_text: str | None = None, user: str | None = None) -> str:
        run_id = f"{kind}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        self._insert("eval_runs", {"id": run_id, "ts": now_iso(), "kind": kind, "label": label,
                                   "config": config, "prompt_version": prompt_version,
                                   "prompt_text": prompt_text, "summary": summary, "user": user})
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO eval_cases (run_id, case_id, question, data) VALUES (?, ?, ?, ?)",
                [(run_id, str(c.get("id") or i), c.get("question") or c.get("text") or c.get("sql"),
                  json.dumps(c, ensure_ascii=False, default=str)) for i, c in enumerate(cases)],
            )
        return run_id

    def list_eval_runs(self, kind: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if kind:
            return self.query("SELECT id, ts, kind, label, config, prompt_version, summary, user "
                              "FROM eval_runs WHERE kind = ? ORDER BY ts DESC, rowid DESC LIMIT ?", (kind, limit))
        return self.query("SELECT id, ts, kind, label, config, prompt_version, summary, user "
                          "FROM eval_runs ORDER BY ts DESC, rowid DESC LIMIT ?", (limit,))

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        runs = self.query("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
        if not runs:
            return None
        run = runs[0]
        run["cases"] = [c["data"] for c in self.query(
            "SELECT data FROM eval_cases WHERE run_id = ? ORDER BY rowid", (run_id,))]
        return run

    def latest_eval_run(self, kind: str) -> dict[str, Any] | None:
        rows = self.query("SELECT id FROM eval_runs WHERE kind = ? ORDER BY ts DESC, rowid DESC LIMIT 1", (kind,))
        return self.get_eval_run(rows[0]["id"]) if rows else None

    # ----------------------------------------------------------------- drift
    def save_drift_report(self, **row: Any) -> int:
        row.setdefault("ts", now_iso())
        return self._insert("drift_reports", row)

    def drift_reports(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM drift_reports ORDER BY ts DESC, rowid DESC LIMIT ?", (limit,))

    # ---------------------------------------------------------------- alerts
    def open_alert(self, rule: str, severity: str, message: str) -> bool:
        """Apre un alert se per quella regola non ce n'è già uno aperto. True se nuovo."""
        if self.query("SELECT id FROM alerts WHERE rule = ? AND resolved_at IS NULL", (rule,)):
            return False
        self._insert("alerts", {"ts": now_iso(), "rule": rule, "severity": severity, "message": message})
        return True

    def resolve_alert(self, rule: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("UPDATE alerts SET resolved_at = ? WHERE rule = ? AND resolved_at IS NULL",
                               (now_iso(), rule))
            return cur.rowcount > 0

    def alerts(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM alerts ORDER BY ts DESC, rowid DESC LIMIT ?", (limit,))

    # -------------------------------------------------------- model registry
    def register_model(self, **row: Any) -> None:
        row.setdefault("ts", now_iso())
        self._insert("model_registry", row)

    def models(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM model_registry ORDER BY ts DESC, rowid DESC")

    def production_model(self) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM model_registry WHERE status = 'production' LIMIT 1")
        return rows[0] if rows else None

    def set_production(self, version: str, user: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE model_registry SET status = 'archived' WHERE status = 'production'")
            conn.execute("UPDATE model_registry SET status = 'production', promoted_at = ?, promoted_by = ? "
                         "WHERE version = ?", (now_iso(), user, version))


_stores: dict[Path, AppStore] = {}
_stores_lock = threading.Lock()


def get_store(path: Path) -> AppStore:
    """Un AppStore per file (crea lo schema una sola volta)."""
    with _stores_lock:
        if path not in _stores:
            _stores[path] = AppStore(path)
        return _stores[path]
