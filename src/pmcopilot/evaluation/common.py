"""
common.py — funzioni condivise dalla suite di valutazione.

LE QUATTRO DOMANDE DELLA VALUTAZIONE
    1. Il MODELLO ML prevede bene i guasti?                  (ml/train.py, kind "model")
    2. Il RETRIEVAL trova il documento giusto?                (retrieval.py)
    3. I GUARDRAIL bloccano gli attacchi e lasciano passare
       le domande legittime?                                  (guardrails.py)
    4. L'AGENTE sceglie i tool giusti e risponde in modo
       fedele ai dati?                                        (agent.py)

LO STORICO: OGNI RUN È UN ESPERIMENTO
    Ogni esecuzione salva in app.db (tabelle eval_runs / eval_cases):
      * la CONFIGURAZIONE completa: modello LLM, giudice, versione e testo dei
        prompt, modalità di retrieval, chunk size, versione del modello ML,
        commit git. È ciò che permette di rispondere a "cosa è cambiato?";
      * le metriche riassuntive;
      * il risultato di OGNI caso, per il confronto domanda per domanda.
    Niente viene sovrascritto: la pagina Esperimenti confronta due run qualsiasi.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from pmcopilot.agent.prompts import PROMPT_VERSION
from pmcopilot.config import PROJECT_ROOT, Settings


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip() and not line.startswith("//")]


def dataset_path(settings: Settings, name: str) -> Path:
    return settings.eval_dir / "datasets" / name


def git_commit() -> str | None:
    """Commit corrente del repository (se è un repo git), per legare un run al codice."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,  # noqa: S603, S607
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:
        return None


def model_version(settings: Settings) -> str | None:
    if settings.model_card_path.exists():
        return json.loads(settings.model_card_path.read_text()).get("model_version")
    return None


def run_config(settings: Settings, **extra: Any) -> dict[str, Any]:
    """Fotografia della configurazione che influenza i risultati."""
    return {
        "llm": settings.llm_spec,
        "judge": settings.effective_judge_model,
        "temperature": settings.llm_temperature,
        "prompt_version": PROMPT_VERSION,
        "retriever_mode": settings.retriever_mode,
        "retriever_k": settings.retriever_k,
        "chunk_size": settings.chunk_size,
        "embedding": f"{settings.embedding_provider}:{settings.embedding_model}",
        "router": settings.router_enabled,
        "injection_classifier": settings.injection_classifier,
        "ml_model": model_version(settings),
        "git_commit": git_commit(),
        **extra,
    }


def write_markdown_report(settings: Settings) -> Path:
    """Report leggibile con l'ultimo run di ogni tipo (lo storico completo è in app.db)."""
    from pmcopilot.store import get_store

    store = get_store(settings.app_db_path)
    parts = ["# Evaluation report\n"]
    for kind, title in (("model", "Modello ML"), ("retrieval", "Retrieval (RAG)"),
                        ("guardrails", "Guardrail"), ("agent", "Agente end-to-end")):
        run = store.latest_eval_run(kind)
        if not run:
            parts.append(f"## {title}\n_Nessun run: `pmc eval {kind}`_\n")
            continue
        rows = "\n".join(f"| {k} | {v} |" for k, v in run["summary"].items())
        parts.append(f"## {title}\nRun `{run['id']}` del {run['ts']}\n\n| Metrica | Valore |\n|---|---|\n{rows}\n")
    report = settings.reports_dir / "eval_report.md"
    report.write_text("\n".join(parts))
    return report
