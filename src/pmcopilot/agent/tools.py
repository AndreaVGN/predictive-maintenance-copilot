"""
tools.py — gli "strumenti" che l'agente può usare.

COS'È UN TOOL
    Una funzione Python + una descrizione in linguaggio naturale + uno schema
    degli argomenti. L'LLM non esegue codice: legge le descrizioni, decide
    QUALE tool chiamare e con QUALI argomenti (tool calling), e LangGraph
    esegue davvero la funzione. Il risultato torna all'LLM come messaggio.

    Per questo le descrizioni (docstring) sono importantissime: sono il
    "manuale d'uso" che il modello legge. Una descrizione vaga = tool usato male.

I CINQUE TOOL
    sql_query        interroga il database in sola lettura (text-to-SQL)
    predict_failure  usa il modello sklearn sull'ultima lettura di una macchina
    search_manual    RAG sui manuali (ricerca ibrida, filtro per modalità di guasto)
    make_chart       crea un grafico matplotlib dell'andamento di una variabile
    open_work_order  apre un ordine di lavoro -> RICHIEDE APPROVAZIONE UMANA

CONTENT E ARTIFACT
    Con response_format="content_and_artifact" un tool restituisce due cose:
      * content  -> testo che l'LLM legge;
      * artifact -> dati strutturati che l'LLM NON vede ma il nostro codice sì
                    (fonti, nome del grafico, SQL eseguito). Il guardrail di
                    uscita e l'API li leggono dai ToolMessage.

ERRORI COME TESTO
    Se una query è sbagliata non solleviamo eccezioni: restituiamo "ERRORE: ..."
    con la spiegazione. L'agente legge l'errore e riprova correggendosi.
    È uno dei cicli che rendono "agentico" il sistema.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import BaseTool, tool

from pmcopilot import domain
from pmcopilot.agent.guardrails import SQLGuardError, sanitize_note, validate_sql
from pmcopilot.analysis.style import COLOR_FAILURE, COLOR_NORMAL, apply_style
from pmcopilot.config import Settings
from pmcopilot.data.database import READABLE_TABLES, connect, connect_readonly
from pmcopilot.ml.predictor import FailurePredictor, MachineReading
from pmcopilot.rag.retriever import ManualRetriever, format_chunks_for_llm

MAX_TOOL_OUTPUT_CHARS = 4000  # oltre, il risultato riempirebbe il contesto dell'LLM

ChartMetric = Literal[
    "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min", "failure_proba"
]
METRIC_LABELS = {
    "air_temp_k": "Temperatura aria [K]",
    "process_temp_k": "Temperatura processo [K]",
    "rpm": "Velocità [rpm]",
    "torque_nm": "Coppia [Nm]",
    "tool_wear_min": "Usura utensile [min]",
    "failure_proba": "Probabilità di guasto",
}


@dataclass
class ToolContext:
    """Tutto ciò di cui i tool hanno bisogno, preparato una volta all'avvio."""

    settings: Settings
    predictor: FailurePredictor
    retriever: ManualRetriever
    machine_ids: set[str]


def _run_readonly_query(settings: Settings, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Esegue una query già validata, con timeout, su una connessione read-only.

    I valori variabili passano SEMPRE come parametri "?" (query parametrizzate):
    è la difesa standard contro la SQL injection.
    """
    with closing(connect_readonly(settings.db_path)) as conn:
        deadline = time.monotonic() + settings.sql_timeout_s
        # Il progress handler viene chiamato ogni N istruzioni della VM di SQLite:
        # se restituisce un valore "vero", la query viene interrotta.
        conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
        return pd.read_sql_query(sql, conn, params=params)


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return text[:MAX_TOOL_OUTPUT_CHARS] + "\n... [output troncato: usa filtri o aggregazioni]"


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    """Crea i tool come closure: ognuno 'ricorda' il contesto ctx."""
    settings = ctx.settings

    def _check_machine(machine_id: str) -> str | None:
        """Restituisce un messaggio di errore se il codice macchina non esiste."""
        if machine_id.upper() not in ctx.machine_ids:
            return (f"ERRORE: la macchina '{machine_id}' non esiste. "
                    f"Codici validi: {', '.join(sorted(ctx.machine_ids))}.")
        return None

    # ------------------------------------------------------------- sql_query
    @tool(response_format="content_and_artifact")
    def sql_query(query: str) -> tuple[str, dict]:
        """Esegue UNA query SQL SELECT (dialetto SQLite) sul database di stabilimento
        e restituisce le righe in formato CSV.

        Usalo per domande su dati storici e attuali: letture dei sensori, guasti,
        interventi di manutenzione, rischio stimato, ordini di lavoro.
        Regole: solo SELECT; massimo 200 righe; preferisci aggregazioni
        (COUNT, AVG, GROUP BY) e le viste v_latest_reading, v_failures_by_machine,
        v_machine_risk. I timestamp sono testi 'YYYY-MM-DD HH:MM:SS'.
        Se ricevi un ERRORE, correggi la query e riprova.
        """
        try:
            safe_sql = validate_sql(query, READABLE_TABLES, settings.sql_max_rows)
        except SQLGuardError as exc:
            return f"ERRORE (guardrail SQL): {exc}", {"sql": query, "blocked": True}
        try:
            df = _run_readonly_query(settings, safe_sql)
        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            return f"ERRORE durante l'esecuzione: {exc}", {"sql": safe_sql, "blocked": False}
        if df.empty:
            return "La query non ha restituito righe.", {"sql": safe_sql, "rows": 0}
        csv = df.to_csv(index=False, float_format="%.4g")
        return _truncate(f"{len(df)} righe\n{csv}"), {"sql": safe_sql, "rows": len(df)}

    # ------------------------------------------------------- predict_failure
    @tool(response_format="content_and_artifact")
    def predict_failure(
        machine_id: str,
        torque_nm: float | None = None,
        rpm: float | None = None,
        tool_wear_min: float | None = None,
        process_temp_k: float | None = None,
    ) -> tuple[str, dict]:
        """Stima con il modello di machine learning la probabilità di guasto di una
        macchina, usando la sua ULTIMA lettura dei sensori.

        Gli argomenti opzionali (coppia, giri, usura, temperatura di processo)
        sovrascrivono i valori dell'ultima lettura: usali per scenari "cosa
        succede se..." (what-if). Restituisce probabilità, livello di rischio
        (basso/medio/alto) e i fattori di rischio fisici superati o vicini al limite.
        """
        machine_id = machine_id.upper().strip()
        if error := _check_machine(machine_id):
            return error, {}
        latest = _run_readonly_query(
            settings,
            "SELECT r.*, m.machine_type FROM v_latest_reading r "
            "JOIN machines m ON m.machine_id = r.machine_id WHERE r.machine_id = ?",
            (machine_id,),
        ).iloc[0]
        overrides = {
            k: v for k, v in {"torque_nm": torque_nm, "rpm": rpm, "tool_wear_min": tool_wear_min,
                              "process_temp_k": process_temp_k}.items() if v is not None
        }
        reading = MachineReading(
            machine_type=str(latest["machine_type"]),
            air_temp_k=float(latest["air_temp_k"]),
            process_temp_k=float(overrides.get("process_temp_k", latest["process_temp_k"])),
            rpm=float(overrides.get("rpm", latest["rpm"])),
            torque_nm=float(overrides.get("torque_nm", latest["torque_nm"])),
            tool_wear_min=float(overrides.get("tool_wear_min", latest["tool_wear_min"])),
        )
        result = ctx.predictor.predict_one(reading)
        scenario = "scenario what-if" if overrides else "ultima lettura"
        factors = "\n".join(f"- {f}" for f in result.risk_factors) or "- nessuna soglia superata o vicina"
        content = (
            f"Macchina {machine_id} (tipo {reading.machine_type}), {scenario} del {latest['ts']}\n"
            f"Valori: aria {reading.air_temp_k} K, processo {reading.process_temp_k} K, "
            f"{reading.rpm:.0f} rpm, {reading.torque_nm} Nm, usura {reading.tool_wear_min:.0f} min\n"
            f"Derivati: delta termico {result.derived['temp_diff_k']} K, potenza "
            f"{result.derived['power_w']} W, usura x coppia {result.derived['wear_x_torque']}\n"
            f"Probabilità di guasto: {result.failure_proba:.1%} -> rischio {result.risk_level.upper()} "
            f"(soglia {result.threshold:.2f}, modello {result.model_version})\n"
            f"Fattori di rischio:\n{factors}"
        )
        return content, {"machine_id": machine_id, "overrides": overrides, **result.to_dict()}

    # --------------------------------------------------------- search_manual
    @tool(response_format="content_and_artifact")
    def search_manual(
        query: str,
        failure_mode: Literal["TWF", "HDF", "PWF", "OSF", "RNF"] | None = None,
    ) -> tuple[str, dict]:
        """Cerca nei manuali di manutenzione e sicurezza dello stabilimento (RAG).

        Usalo per: cause e sintomi dei guasti (TWF, HDF, PWF, OSF, RNF), soglie
        operative, procedure di intervento, codici di allarme (AL-xxx), sicurezza
        (LOTO, DPI), priorità e regole degli ordini di lavoro.
        Scrivi una query specifica in italiano. Se la domanda riguarda UNA modalità
        di guasto, passa failure_mode: la ricerca si limita ai documenti su quel
        guasto più quelli generali (sicurezza, procedure, codici).
        Nella risposta cita le fonti con i numeri [1], [2]...
        """
        chunks = ctx.retriever.search(query, failure_mode=failure_mode)
        return format_chunks_for_llm(chunks), {"sources": [c.citation for c in chunks],
                                               "chunks": [c.to_dict() for c in chunks],
                                               "filter": {"failure_mode": failure_mode},
                                               "mode": ctx.retriever.mode}

    # ------------------------------------------------------------ make_chart
    @tool(response_format="content_and_artifact")
    def make_chart(machine_id: str, metric: ChartMetric, last_hours: int = 48) -> tuple[str, dict]:
        """Crea un grafico (PNG) dell'andamento nel tempo di una variabile per una
        macchina, nelle ultime `last_hours` ore di dati (massimo 504 = 21 giorni).

        metric: air_temp_k, process_temp_k, rpm, torque_nm, tool_wear_min oppure
        failure_proba (probabilità stimata dal modello). I guasti avvenuti sono
        evidenziati. Restituisce il nome del file del grafico.
        """
        machine_id = machine_id.upper().strip()
        if error := _check_machine(machine_id):
            return error, {}
        last_hours = max(1, min(int(last_hours), 504))
        source = "predictions" if metric == "failure_proba" else "sensor_readings"
        failure_col = "0 AS machine_failure" if metric == "failure_proba" else "machine_failure"
        # Nomi di colonna e tabella non possono essere parametri "?": qui sono
        # sicuri perché vengono da un Literal (whitelist), non da testo libero.
        df = _run_readonly_query(
            settings,
            f"SELECT ts, {metric} AS value, {failure_col} FROM {source} "
            f"WHERE machine_id = ? AND ts >= datetime((SELECT MAX(ts) FROM {source}), ?) ORDER BY ts",
            (machine_id, f"-{last_hours} hours"),
        )
        if df.empty:
            return f"Nessun dato di {metric} per {machine_id} nelle ultime {last_hours} ore.", {}
        df["ts"] = pd.to_datetime(df["ts"])

        apply_style()
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(df["ts"], df["value"], color=COLOR_NORMAL, marker="o", markersize=3, linewidth=1.5)
        failures = df[df["machine_failure"] == 1]
        if not failures.empty:
            ax.scatter(failures["ts"], failures["value"], s=70, color=COLOR_FAILURE, zorder=3,
                       edgecolor="white", linewidth=1.2, label="guasto")
            ax.legend(loc="upper left")
        if metric == "tool_wear_min":
            ax.axhline(domain.TOOL_WEAR_REPLACEMENT_MIN, color=COLOR_FAILURE, linestyle="--", linewidth=1)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{machine_id} — {METRIC_LABELS[metric]} (ultime {last_hours} h di dati)")
        fig.autofmt_xdate()
        filename = f"{machine_id}_{metric}_{uuid.uuid4().hex[:8]}.png"
        settings.charts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.charts_dir / filename)
        plt.close(fig)

        stats = df["value"].describe()
        content = (
            f"Grafico creato: {filename} ({len(df)} punti, {len(failures)} guasti evidenziati). "
            f"Statistiche: min {stats['min']:.4g}, media {stats['mean']:.4g}, max {stats['max']:.4g}, "
            f"ultimo valore {df['value'].iloc[-1]:.4g}."
        )
        return content, {"chart": filename}

    # ------------------------------------------------------- open_work_order
    @tool(response_format="content_and_artifact")
    def open_work_order(
        machine_id: str,
        failure_type: Literal["TWF", "HDF", "PWF", "OSF", "RNF", "GENERICO"],
        priority: Literal["P1", "P2", "P3"],
        notes: str,
    ) -> tuple[str, dict]:
        """Apre un ordine di lavoro di manutenzione per una macchina.

        Usalo SOLO se l'utente chiede esplicitamente di aprire un ordine di lavoro.
        Un responsabile umano deve approvare l'ordine prima che venga registrato:
        il sistema chiederà l'approvazione automaticamente.
        Priorità: P1 guasto in corso / oltre soglia; P2 rischio alto; P3 preventiva.
        """
        machine_id = machine_id.upper().strip()
        if error := _check_machine(machine_id):
            return error, {}
        # Unico punto in cui l'agente SCRIVE: con una connessione dedicata,
        # parametri "?" (mai f-string con input esterni!) e dopo l'approvazione.
        # Le note vengono ripulite PRIMA di salvarle: se contenessero istruzioni
        # nascoste, tornerebbero all'agente la prossima volta che legge la tabella
        # (prompt injection indiretta). Chi ha approvato lo registra service.py.
        notes = sanitize_note(notes)
        with closing(connect(settings.db_path)) as conn:
            cursor = conn.execute(
                "INSERT INTO work_orders (machine_id, created_at, failure_type, priority, notes) "
                "VALUES (?, ?, ?, ?, ?)",
                (machine_id, datetime.now(UTC).isoformat(sep=" ", timespec="seconds"),
                 failure_type, priority, notes),
            )
            conn.commit()
            work_order_id = cursor.lastrowid
        return (f"Ordine di lavoro #{work_order_id} registrato per {machine_id} "
                f"({failure_type}, priorità {priority})."), {"work_order_id": work_order_id}

    return [sql_query, predict_failure, search_manual, make_chart, open_work_order]
