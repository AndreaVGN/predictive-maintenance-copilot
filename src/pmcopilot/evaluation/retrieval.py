"""
retrieval.py — valutazione del retrieval con metriche di Information Retrieval.

METRICHE
    hit@1   la fonte giusta è al PRIMO posto?
    hit@k   la fonte giusta è tra i primi k (quelli che l'agente legge davvero)?
    MRR     Mean Reciprocal Rank: media di 1/posizione del primo risultato giusto
            (1 se primo, 0,5 se secondo, 0,33 se terzo... 0 se assente).
            Premia chi mette il documento giusto IN ALTO.

    Le domande del golden set hanno un "type": semantica (parafrasi), numerica
    (soglie come "8,6 K") e codice (allarmi "AL-207"). Le metriche per tipo
    mostrano DOVE una modalità vince: gli embedding sulle parafrasi, BM25 su
    codici e numeri, l'ibrida su entrambi.

CONFRONTO TRA MODALITÀ
    `pmc eval retrieval --mode all` valuta dense, bm25 e hybrid sulle stesse
    domande e salva tre run: nella pagina Esperimenti li confronti affiancati.
"""

from __future__ import annotations

import logging

import numpy as np

from pmcopilot.agent.llm import get_embeddings
from pmcopilot.config import Settings
from pmcopilot.evaluation.common import dataset_path, load_jsonl, run_config
from pmcopilot.rag.retriever import ManualRetriever
from pmcopilot.store import get_store

logger = logging.getLogger(__name__)
MODES = ("dense", "bm25", "hybrid")


def _metrics(ranks: list[int | None], k: int) -> dict[str, float]:
    reciprocal = [1 / r if r else 0.0 for r in ranks]
    return {"hit@1": round(float(np.mean([r == 1 for r in ranks])), 3),
            f"hit@{k}": round(float(np.mean([r is not None for r in ranks])), 3),
            "mrr": round(float(np.mean(reciprocal)), 3)}


def evaluate_retrieval(settings: Settings, mode: str | None = None, label: str | None = None,
                       user: str = "system") -> dict:
    mode = mode or settings.retriever_mode
    k = settings.retriever_k
    retriever = ManualRetriever(settings, get_embeddings(settings), mode=mode)
    cases = load_jsonl(dataset_path(settings, "retrieval_golden.jsonl"))
    details = []
    for i, case in enumerate(cases):
        results = retriever.search(case["question"], k=k)
        sources = [r.source for r in results]
        rank = sources.index(case["expected_source"]) + 1 if case["expected_source"] in sources else None
        details.append({"id": f"r{i + 1:02d}", "question": case["question"], "type": case.get("type", "semantica"),
                        "expected": case["expected_source"], "retrieved": sources, "rank": rank,
                        "passed": rank is not None})
    summary = {"n_questions": len(cases), **_metrics([d["rank"] for d in details], k)}
    for qtype in sorted({d["type"] for d in details}):
        sub = [d["rank"] for d in details if d["type"] == qtype]
        summary[f"mrr_{qtype}"] = _metrics(sub, k)["mrr"]
    run_id = get_store(settings.app_db_path).save_eval_run(
        "retrieval", summary, details, config=run_config(settings, retriever_mode=mode),
        label=label or f"retrieval {mode}", user=user)
    logger.info("Retrieval %s: %s (run %s)", mode, summary, run_id)
    return {"run_id": run_id, **summary}
