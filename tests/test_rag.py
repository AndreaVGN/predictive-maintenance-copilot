"""
Test del RAG: tokenizzazione BM25, fusione RRF, ricerca ibrida, filtri sui
metadati e reindicizzazione incrementale.

Con gli embedding finti (hash del testo) la ricerca densa è casuale: i test
sulla QUALITÀ della ricerca usano quindi BM25, che è deterministico. La qualità
della ricerca densa con nomic-embed-text vero si misura con `pmc eval retrieval`.
"""

import shutil

import pytest

from pmcopilot.agent.llm import get_embeddings
from pmcopilot.rag.bm25 import BM25Index, light_stem, reciprocal_rank_fusion, tokenize
from pmcopilot.rag.ingest import build_vector_store, manuals_changed, parse_front_matter
from pmcopilot.rag.retriever import ManualRetriever


# ------------------------------------------------------------------ BM25
def test_tokenize_strips_accents_stopwords_and_keeps_codes():
    tokens = tokenize("Qual è la temperatura dell'aria? Allarme AL-207, 8,6 K")
    assert "al-207" in tokens and "207" in tokens  # codice intero più le sue parti
    assert "8,6" in tokens
    assert "la" not in tokens and "l" not in tokens  # stopword e lettere singole fuori
    assert light_stem("temperature") == light_stem("temperatura")  # singolare e plurale coincidono


def test_bm25_ranks_exact_term_first():
    docs = ["la coppia del mandrino", "allarme AL-207 ventola bloccata", "usura utensile e sostituzione"]
    scores = BM25Index(docs).scores("cosa significa AL-207")
    assert scores.index(max(scores)) == 1
    assert BM25Index(docs).scores("parola assente") == [0.0, 0.0, 0.0]


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60))
    # "a" e "b" sono in alto in entrambe le classifiche: battono "c" e "d".
    assert min(fused["a"], fused["b"]) > max(fused["c"], fused["d"])
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)


# ------------------------------------------------------------ retriever
def test_bm25_mode_finds_alarm_code(tool_ctx):
    retriever = tool_ctx.retriever
    chunks = retriever.search("Cosa significa l'allarme AL-207?", k=3, mode="bm25")
    assert chunks[0].source == "08_codici_allarme.md"


def test_hybrid_returns_k_unique_chunks(tool_ctx):
    chunks = tool_ctx.retriever.search("guasto per dissipazione termica", k=4, mode="hybrid")
    assert len(chunks) == 4
    assert len({c.chunk_id for c in chunks}) == 4


def test_failure_mode_filter_keeps_general_documents(tool_ctx):
    """Con il filtro HDF restano i documenti HDF e quelli generali ("*"), mai TWF/PWF/OSF."""
    for mode in ["dense", "bm25", "hybrid"]:
        chunks = tool_ctx.retriever.search("cosa controllare", k=6, failure_mode="HDF", mode=mode)
        assert chunks, mode
        assert {c.failure_mode for c in chunks} <= {"HDF", "*"}, mode
    # Un valore non valido non deve azzerare i risultati: il filtro viene ignorato.
    assert tool_ctx.retriever.search("usura", k=2, failure_mode="XYZ", mode="bm25")


def test_doc_type_filter(tool_ctx):
    chunks = tool_ctx.retriever.search("procedura", k=5, doc_type="procedura", mode="bm25")
    assert chunks and {c.doc_type for c in chunks} == {"procedura"}


# ----------------------------------------------------------- indicizzazione
def test_parse_front_matter():
    meta, body = parse_front_matter('---\ndoc_type: guasto\nfailure_mode: "HDF"\n---\n# Titolo\ntesto')
    assert meta == {"doc_type": "guasto", "failure_mode": "HDF"}
    assert body.startswith("# Titolo")
    assert parse_front_matter("# Senza metadati")[0] == {}


def test_incremental_reindex(env, tmp_path):
    """Solo i manuali cambiati vengono re-indicizzati; quelli cancellati spariscono dall'indice."""
    manuals = tmp_path / "manuals"
    shutil.copytree(env.manuals_dir, manuals)
    settings = env.model_copy(update={"data_dir": tmp_path / "data", "manuals_dir": manuals})
    settings.ensure_dirs()
    emb = get_embeddings(settings)

    first = build_vector_store(settings, emb)
    assert first["added"] == len(list(manuals.glob("*.md"))) and first["unchanged"] == 0
    assert not manuals_changed(settings)

    again = build_vector_store(settings, emb)
    assert again["added"] == again["updated"] == again["removed"] == 0

    (manuals / "03_pwf_potenza.md").write_text((manuals / "03_pwf_potenza.md").read_text() + "\n\nNota: parola-test-xyz.\n")
    (manuals / "05_rnf_guasti_casuali.md").unlink()
    assert manuals_changed(settings)
    diff = build_vector_store(settings, emb)
    assert diff["updated"] == 1 and diff["removed"] == 1

    retriever = ManualRetriever(settings, emb)
    assert retriever.search("parola-test-xyz", k=1, mode="bm25")[0].source == "03_pwf_potenza.md"
    assert all(c.source != "05_rnf_guasti_casuali.md" for c in retriever.search("guasti casuali", k=10, mode="bm25"))
