"""
retriever.py — la "R" del RAG: data una domanda, trova i chunk più utili.

TRE MODALITÀ (PMC_RETRIEVER_MODE), confrontabili nella pagina Esperimenti
    dense   similarità semantica tra embedding (distanza coseno, indice HNSW)
    bm25    corrispondenza di parole chiave (vedi bm25.py)
    hybrid  entrambe, fuse con Reciprocal Rank Fusion (default)

METADATA FILTERING
    Prima di cercare si può restringere il campo: failure_mode="HDF" cerca solo
    nei documenti su HDF più quelli generali (failure_mode "*": sicurezza,
    codici di allarme, procedure). Meno candidati sbagliati = meno rumore
    nel contesto dell'LLM. Il filtro è applicato a ENTRAMBE le ricerche:
    in Chroma con una clausola `where`, in BM25 scartando i chunk non ammessi.

IL FLUSSO DELLA RICERCA IBRIDA
    domanda ──► embedding ──► Chroma (top 3k, con filtro) ──┐
           └──► token ─────► BM25 (top 3k, con filtro) ────┴─► RRF ─► top k
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from pmcopilot.config import Settings
from pmcopilot.rag.bm25 import BM25Index, reciprocal_rank_fusion
from pmcopilot.rag.ingest import COLLECTION_NAME

FAILURE_MODES = ("TWF", "HDF", "PWF", "OSF", "RNF")


@dataclass
class RetrievedChunk:
    source: str
    section: str
    content: str
    score: float  # dense: distanza coseno (più bassa = meglio); bm25/hybrid: punteggio (più alto = meglio)
    doc_type: str = ""
    failure_mode: str = ""
    chunk_id: str = ""

    @property
    def citation(self) -> str:
        return f"{self.source} › {self.section}" if self.section else self.source

    def to_dict(self) -> dict:
        return {**asdict(self), "citation": self.citation}


class ManualRetriever:
    """Incapsula vector store e indice BM25: il resto del codice non dipende da Chroma."""

    def __init__(self, settings: Settings, embeddings: Embeddings, mode: str | None = None):
        if not settings.chroma_dir.exists():
            raise FileNotFoundError(f"Indice non trovato in {settings.chroma_dir}. Esegui 'pmc ingest'.")
        self.settings = settings
        self.k = settings.retriever_k
        self.mode = mode or settings.retriever_mode
        self.store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
                            persist_directory=str(settings.chroma_dir),
                            create_collection_if_not_exists=False)
        self._lock = threading.Lock()
        self._bm25_count = -1
        self._refresh_bm25()

    # ------------------------------------------------------------- BM25
    def _refresh_bm25(self) -> None:
        """(Ri)costruisce l'indice BM25 dai chunk in Chroma, se il loro numero è cambiato."""
        count = self.store._collection.count()
        if count == self._bm25_count:
            return
        with self._lock:
            data = self.store.get(include=["documents", "metadatas"])
            self._ids = data["ids"]
            self._texts = data["documents"]
            self._metas = data["metadatas"]
            self._bm25 = BM25Index(self._texts)
            self._bm25_count = count

    # ----------------------------------------------------------- filtri
    @staticmethod
    def _where(failure_mode: str | None, doc_type: str | None) -> dict | None:
        clauses = []
        if failure_mode:
            clauses.append({"failure_mode": {"$in": [failure_mode.upper(), "*"]}})
        if doc_type:
            clauses.append({"doc_type": doc_type})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _allowed(meta: dict, failure_mode: str | None, doc_type: str | None) -> bool:
        if failure_mode and meta.get("failure_mode") not in (failure_mode.upper(), "*"):
            return False
        return not (doc_type and meta.get("doc_type") != doc_type)

    def _chunk(self, meta: dict, text: str, score: float) -> RetrievedChunk:
        return RetrievedChunk(source=meta.get("source", "?"), section=meta.get("section", ""),
                              content=text, score=round(float(score), 4),
                              doc_type=meta.get("doc_type", ""), failure_mode=meta.get("failure_mode", ""),
                              chunk_id=meta.get("chunk_id", ""))

    # ---------------------------------------------------------- ricerche
    def _dense(self, query: str, n: int, where: dict | None) -> list[RetrievedChunk]:
        results = self.store.similarity_search_with_score(query, k=n, filter=where)
        return [self._chunk(doc.metadata, doc.page_content, dist) for doc, dist in results]

    def _bm25_search(self, query: str, n: int, failure_mode: str | None,
                     doc_type: str | None) -> list[RetrievedChunk]:
        self._refresh_bm25()
        scored = [(s, i) for i, s in enumerate(self._bm25.scores(query))
                  if s > 0 and self._allowed(self._metas[i], failure_mode, doc_type)]
        scored.sort(reverse=True)
        return [self._chunk(self._metas[i], self._texts[i], s) for s, i in scored[:n]]

    def search(self, query: str, k: int | None = None, *, failure_mode: str | None = None,
               doc_type: str | None = None, mode: str | None = None) -> list[RetrievedChunk]:
        k = k or self.k
        mode = mode or self.mode
        if failure_mode and failure_mode.upper() not in FAILURE_MODES:
            failure_mode = None  # valore non valido: meglio nessun filtro che zero risultati
        if mode == "dense":
            return self._dense(query, k, self._where(failure_mode, doc_type))
        if mode == "bm25":
            return self._bm25_search(query, k, failure_mode, doc_type)
        # hybrid: più candidati da ciascuna ricerca, poi fusione per posizione.
        n = k * 3
        dense = self._dense(query, n, self._where(failure_mode, doc_type))
        sparse = self._bm25_search(query, n, failure_mode, doc_type)
        by_id = {c.chunk_id: c for c in dense + sparse}
        fused = reciprocal_rank_fusion([[c.chunk_id for c in dense], [c.chunk_id for c in sparse]],
                                       k=self.settings.rrf_k)
        out = []
        for chunk_id, score in fused[:k]:
            chunk = by_id[chunk_id]
            chunk.score = round(score, 5)
            out.append(chunk)
        return out


def format_chunks_for_llm(chunks: list[RetrievedChunk]) -> str:
    """Testo che l'agente legge: chunk numerati con la loro fonte."""
    if not chunks:
        return "Nessun documento pertinente trovato nei manuali."
    blocks = [f"[{i}] (fonte: {c.citation})\n{c.content}" for i, c in enumerate(chunks, start=1)]
    return "\n\n---\n\n".join(blocks)
