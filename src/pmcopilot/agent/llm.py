"""
llm.py — factory per i modelli di chat e di embedding, locali (Ollama) o cloud.

PERCHÉ UNA FACTORY
    Il resto del codice chiede "dammi il modello" e non sa se dietro c'è
    Ollama sul Mac, un'API cloud o un modello finto per i test. Cambiare
    provider = cambiare due righe nel .env:

        PMC_LLM_PROVIDER=ollama      PMC_LLM_MODEL=qwen3:8b          (default)
        PMC_LLM_PROVIDER=openai      PMC_LLM_MODEL=<modello OpenAI>   + OPENAI_API_KEY
        PMC_LLM_PROVIDER=anthropic   PMC_LLM_MODEL=<modello Claude>   + ANTHROPIC_API_KEY

    Il grafo, i tool, i guardrail e la valutazione restano identici: è proprio
    questo che permette di CONFRONTARE i modelli sulla stessa suite di test.

I PARAMETRI DI SICUREZZA DEL MODELLO
    max_output_tokens  tetto ai token generati per chiamata (un modello che
                       "divaga" non può riempire il contesto né il conto)
    timeout            una chiamata bloccata non tiene appeso l'utente per sempre
    retry              li gestisce ModelRetryMiddleware (graph.py) in modo uguale
                       per tutti i provider: per questo qui disattiviamo i retry
                       interni dei client cloud (max_retries=0), altrimenti i
                       tentativi si moltiplicherebbero (2 x 2 = 4).

PRIVACY
    Con un provider cloud, domande, schema del database e risultati dei tool
    escono dall'azienda. In produzione serve un accordo sul trattamento dei
    dati (DPA) e una valutazione GDPR. Ollama invece tiene tutto in locale.
"""

from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel

from pmcopilot.config import Settings

CLOUD_PACKAGES = {"openai": "langchain-openai", "anthropic": "langchain-anthropic"}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """"ollama:qwen3:8b" -> ("ollama", "qwen3:8b"). Divide solo sul PRIMO ":"."""
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"Specifica modello non valida '{spec}': usa 'provider:modello'.")
    return provider, model


def build_chat_model(settings: Settings, provider: str, model: str) -> BaseChatModel:
    """Crea un modello di chat per il provider indicato, con limiti e timeout."""
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict = {
            "model": model,
            "base_url": settings.ollama_base_url,
            "temperature": settings.llm_temperature,
            "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_max_output_tokens,
            "client_kwargs": {"timeout": settings.llm_timeout_s},
        }
        if settings.llm_reasoning is not None:
            kwargs["reasoning"] = settings.llm_reasoning
        return ChatOllama(**kwargs)

    common = {"model": model, "temperature": settings.llm_temperature,
              "timeout": settings.llm_timeout_s, "max_retries": 0,
              "max_tokens": settings.llm_max_output_tokens}
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(**common)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**common)
    except ImportError as exc:
        raise RuntimeError(f"Per il provider '{provider}' installa: pip install -e \".[cloud]\" "
                           f"(pacchetto {CLOUD_PACKAGES[provider]}).") from exc
    raise ValueError(f"Provider LLM non supportato: {provider}")


def get_chat_model(settings: Settings, spec: str | None = None) -> BaseChatModel:
    """Il modello dell'agente (o quello indicato da `spec`, es. il giudice)."""
    if spec:
        return build_chat_model(settings, *parse_model_spec(spec))
    return build_chat_model(settings, settings.llm_provider, settings.llm_model)


def estimate_cost_usd(settings: Settings, tokens_in: int, tokens_out: int) -> float:
    """Costo stimato di una richiesta con i prezzi del .env (0 per i modelli locali)."""
    return round((tokens_in * settings.llm_price_in_per_mtok
                  + tokens_out * settings.llm_price_out_per_mtok) / 1_000_000, 6)


class NomicPrefixedEmbeddings(Embeddings):
    """Aggiunge i prefissi di task richiesti da nomic-embed-text.

    Il modello è addestrato a distinguere DOCUMENTI e DOMANDE tramite un prefisso
    ("search_document: ..." / "search_query: ..."). Senza prefissi funziona lo
    stesso, ma il retrieval è meno preciso: dettaglio che si trova nella model card.
    """

    def __init__(self, inner: Embeddings):
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(f"search_query: {text}")


def get_embeddings(settings: Settings) -> Embeddings:
    """Embedding per il RAG. Cambiare modello di embedding richiede di reindicizzare."""
    if settings.embedding_provider == "fake":
        # Vettori pseudo-casuali ma deterministici: solo per i test, senza significato.
        return DeterministicFakeEmbedding(size=256)
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)
    from langchain_ollama import OllamaEmbeddings

    base = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
    return NomicPrefixedEmbeddings(base) if "nomic" in settings.embedding_model else base
