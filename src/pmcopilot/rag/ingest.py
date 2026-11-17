"""
ingest.py — indicizzazione dei manuali: documenti -> chunk -> vettori -> Chroma.

LA PIPELINE
    1. LOAD   legge i Markdown di data/manuals/ e la loro "front matter"
              (l'intestazione tra ---: doc_type, failure_mode, revision)
    2. SPLIT  divide per titoli (# e ##), poi per lunghezza con overlap
    3. EMBED  ogni chunk diventa un vettore
    4. STORE  vettori, testo e METADATI in Chroma, su disco

METADATI (servono al filtering)
    source        nome del file              -> citazioni
    title/section titolo e sezione           -> citazioni
    doc_type      guasto, sicurezza, ...      -> filtro per tipo di documento
    failure_mode  TWF/HDF/PWF/OSF/RNF o "*"   -> filtro per modalità di guasto
                  ("*" = documento generale, valido per tutte)
    revision      versione del documento      -> tracciabilità
    doc_hash      impronta del file           -> reindicizzazione incrementale

REINDICIZZAZIONE INCREMENTALE
    Ricalcolare tutti gli embedding a ogni modifica è uno spreco (e con
    un'API cloud costa). Salviamo un "manifest": nome file -> hash SHA-256 del
    contenuto. Alla successiva ingestione:
      * file nuovo o con hash diverso -> si cancellano i suoi vecchi chunk e
        si indicizzano i nuovi;
      * file sparito                  -> si cancellano i suoi chunk;
      * file invariato                -> non si tocca.
    Lo scheduler dell'API lo fa da solo ogni ora: aggiorni un manuale e
    l'assistente lo "sa" senza riavvii.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from pmcopilot.config import Settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "maintenance_manuals"
FRONT_MATTER_KEYS = ("doc_type", "failure_mode", "revision")


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Separa l'intestazione "chiave: valore" tra due righe '---' dal corpo del documento."""
    if not text.startswith("---"):
        return {}, text
    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body.lstrip()


def load_manuals(settings: Settings) -> list[Document]:
    docs = []
    for path in sorted(settings.manuals_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        docs.append(Document(page_content=body, metadata={
            "source": path.name,
            "doc_hash": hashlib.sha256(raw.encode()).hexdigest()[:16],
            "doc_type": meta.get("doc_type", "generico"),
            "failure_mode": meta.get("failure_mode", "*"),
            "revision": meta.get("revision", ""),
        }))
    if not docs:
        raise FileNotFoundError(f"Nessun manuale .md trovato in {settings.manuals_dir}")
    return docs


def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks: list[Document] = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            for piece in size_splitter.split_documents([section]):
                title = piece.metadata.get("title", "")
                sect = piece.metadata.get("section", "")
                header = " › ".join(x for x in (title, sect) if x)
                # "Contextual chunk header": il titolo in testa dà contesto all'embedding.
                text = f"{header}\n\n{piece.page_content}" if header else piece.page_content
                chunk_id = hashlib.sha1(f"{doc.metadata['source']}|{text}".encode()).hexdigest()[:16]
                chunks.append(Document(page_content=text, metadata={
                    **doc.metadata, "title": title, "section": sect, "chunk_id": chunk_id}))
    return chunks


def _open_store(settings: Settings, embeddings: Embeddings) -> Chroma:
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
                  persist_directory=str(settings.chroma_dir),
                  collection_metadata={"hnsw:space": "cosine"})


def _read_manifest(settings: Settings) -> dict[str, str]:
    path = settings.index_manifest_path
    return json.loads(path.read_text()) if path.exists() else {}


def build_vector_store(settings: Settings, embeddings: Embeddings, full: bool = False) -> dict[str, int]:
    """Indicizza i manuali. Incrementale di default; full=True ricostruisce tutto.

    Restituisce quanti documenti sono stati aggiunti, aggiornati, rimossi o lasciati
    invariati, e quanti chunk ci sono nell'indice.
    """
    docs = load_manuals(settings)
    if full and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    manifest = {} if full else _read_manifest(settings)
    if not settings.chroma_dir.exists():
        manifest = {}
    store = _open_store(settings, embeddings)

    current = {d.metadata["source"]: d.metadata["doc_hash"] for d in docs}
    stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}
    for source in set(manifest) - set(current):  # file cancellati
        store.delete(where={"source": source})
        stats["removed"] += 1
    changed = []
    for doc in docs:
        source = doc.metadata["source"]
        if manifest.get(source) == doc.metadata["doc_hash"]:
            stats["unchanged"] += 1
            continue
        if source in manifest:
            store.delete(where={"source": source})
            stats["updated"] += 1
        else:
            stats["added"] += 1
        changed.append(doc)
    if changed:
        chunks = split_documents(changed, settings)
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])

    settings.index_manifest_path.write_text(json.dumps(current, indent=2))
    stats["chunks"] = store._collection.count()
    logger.info("Indicizzazione manuali: %s", stats)
    return stats


def manuals_changed(settings: Settings) -> bool:
    """True se i manuali su disco non corrispondono all'ultimo indice (per lo scheduler)."""
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
               for p in settings.manuals_dir.glob("*.md")}
    return current != _read_manifest(settings)
