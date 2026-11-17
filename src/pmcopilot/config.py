"""
config.py — configurazione centralizzata del progetto.

PERCHÉ ESISTE
    Tutti i moduli (dati, modello, RAG, agente, API) hanno bisogno di sapere
    "dove sono i file", "quale modello usare", "quali limiti applicare".
    Invece di spargere costanti nel codice, le raccogliamo in `Settings`.

COME FUNZIONA
    `pydantic-settings` legge i valori in quest'ordine di priorità:
      1. variabili d'ambiente con prefisso PMC_   (es. PMC_LLM_MODEL=gemma4)
      2. il file .env nella radice del progetto
      3. i valori di default scritti qui sotto
    Pydantic valida i tipi: se scrivi PMC_RETRIEVER_K=ciao, l'avvio fallisce
    subito con un errore chiaro invece di rompersi più avanti ("fail fast").

AMBIENTI (dev / prod)
    PMC_ENV=prod rende il sistema più severo: rifiuta di partire con la
    password di default (a meno di PMC_ALLOW_DEFAULT_PASSWORD=true), usa
    cookie "Secure" (solo HTTPS) e non mostra i dettagli degli errori.

COME SI USA
    from pmcopilot.config import get_settings
    settings = get_settings()

    Nei test creiamo Settings(...) a mano puntando a cartelle temporanee: per
    questo le funzioni ricevono `settings` come parametro (dependency injection).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Radice del repository: questo file sta in src/pmcopilot/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ADMIN_PASSWORD = "admin"


class Settings(BaseSettings):
    """Tutti i parametri configurabili del progetto, con valori di default sensati."""

    model_config = SettingsConfigDict(env_prefix="PMC_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    env: Literal["dev", "prod"] = "dev"

    # ------------------------------------------------------------------ cartelle
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"  # modelli addestrati (registry)
    reports_dir: Path = PROJECT_ROOT / "reports"  # grafici e report
    logs_dir: Path = PROJECT_ROOT / "logs"  # log applicativi e trace
    eval_dir: Path = PROJECT_ROOT / "eval"  # golden set
    manuals_dir: Path = PROJECT_ROOT / "data" / "manuals"  # documenti per il RAG
    backups_dir: Path = PROJECT_ROOT / "backups"

    # ---------------------------------------------------------------------- dati
    random_seed: int = 42
    force_synthetic_data: bool = False  # True = non scaricare AI4I 2020, usa il generatore

    # ----------------------------------------------------------------------- LLM
    # provider: "ollama" (locale, default) oppure un'API cloud: "openai", "anthropic".
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    llm_model: str = "qwen3:8b"  # con un'API cloud: il nome del modello del provider
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 1024  # tetto ai token generati per singola chiamata
    llm_timeout_s: float = 120.0  # oltre, la chiamata fallisce (e scatta il retry)
    llm_max_retries: int = 2  # tentativi extra su errori temporanei (rete, 5xx, 429)
    # Modello di riserva se il principale fallisce anche dopo i retry,
    # nel formato "provider:modello" (es. "ollama:llama3.1:8b"). Vuoto = nessuno.
    llm_fallback: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    llm_num_ctx: int = 8192  # finestra di contesto (Ollama); per le API è quella del modello
    llm_reasoning: bool | None = None  # modelli "thinking" (qwen3): False = più veloce
    # Prezzi in dollari per milione di token, per stimare i costi (0 = gratis/locale).
    llm_price_in_per_mtok: float = 0.0
    llm_price_out_per_mtok: float = 0.0
    # Giudice per la valutazione, "provider:modello" (vuoto = lo stesso dell'agente).
    judge_model: str | None = None
    # Le chiavi delle API cloud NON hanno il prefisso PMC_: le librerie leggono
    # direttamente OPENAI_API_KEY / ANTHROPIC_API_KEY dalle variabili d'ambiente.

    # ---------------------------------------------------------------- embedding
    embedding_provider: Literal["ollama", "openai", "fake"] = "ollama"
    embedding_model: str = "nomic-embed-text"

    # ---------------------------------------------------------------------- RAG
    chunk_size: int = 900
    chunk_overlap: int = 120
    retriever_k: int = 4
    # dense = solo embedding; bm25 = solo parole chiave; hybrid = entrambi fusi con RRF
    retriever_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    rrf_k: int = 60  # costante della Reciprocal Rank Fusion

    # -------------------------------------------------------------------- agente
    router_enabled: bool = True
    injection_classifier: Literal["off", "llm"] = "off"  # classificatore LLM dopo le regole
    max_model_calls: int = 8
    max_tool_calls: int = 10
    max_tokens_per_request: int = 60_000  # budget complessivo (input + output) di una domanda
    summarize_at_fraction: float = 0.7  # riassume la storia oltre il 70% del contesto
    checkpointer: Literal["sqlite", "memory"] = "sqlite"  # memoria conversazioni

    # ----------------------------------------------------------------- guardrail
    max_input_chars: int = 2000
    sql_max_rows: int = 200
    sql_timeout_s: float = 5.0

    # ------------------------------------------------------- autenticazione/limiti
    admin_user: str = "admin"
    admin_password: SecretStr = SecretStr(DEFAULT_ADMIN_PASSWORD)
    operator_user: str | None = None  # utente opzionale che chatta ma NON approva
    operator_password: SecretStr | None = None
    allow_default_password: bool = False  # solo per PMC_ENV=prod
    session_secret: SecretStr | None = None  # firma dei cookie (generato se assente)
    session_hours: int = 12
    rate_limit_per_minute: int = 20  # richieste chat per utente al minuto
    daily_token_budget: int = 2_000_000  # token al giorno per utente (0 = illimitato)

    # ------------------------------------------------------------ observability
    log_level: str = "INFO"
    log_to_file: bool = True
    log_max_mb: int = 20  # rotazione dei file di log e trace
    log_backups: int = 5  # quanti file ruotati conservare (retention)
    trace_to_file: bool = True
    redact_pii: bool = True  # oscura email, telefoni, codici fiscali in log e trace
    langfuse_enabled: bool = False
    langfuse_project_id: str = "pmcopilot"
    alert_webhook_url: str | None = None  # es. webhook Slack/Teams per gli alert
    alert_error_rate: float = 0.05  # alert se più del 5% delle richieste fallisce
    alert_p95_latency_s: float = 60.0
    alert_window_min: int = 15

    # ----------------------------------------------------------- operazioni ML
    scheduler_enabled: bool = True  # job periodici dentro l'API
    scoring_interval_min: int = 60
    drift_interval_min: int = 60
    backup_interval_hours: int = 24
    backups_keep: int = 7
    drift_window_hours: int = 24  # finestra recente confrontata con il training
    drift_psi_warn: float = 0.1  # PSI: < 0.1 stabile, 0.1-0.25 moderato, > 0.25 forte
    drift_psi_alert: float = 0.25

    # ------------------------------------------------- percorsi derivati (property)
    @property
    def raw_csv_path(self) -> Path:
        return self.data_dir / "raw" / "ai4i2020.csv"

    @property
    def data_source_path(self) -> Path:
        return self.data_dir / "raw" / "SOURCE.txt"

    @property
    def db_path(self) -> Path:
        """Database di stabilimento (dati che l'agente può leggere)."""
        return self.data_dir / "processed" / "plant.db"

    @property
    def app_db_path(self) -> Path:
        """Database dell'applicazione (richieste, audit, esperimenti, drift).
        È un file SEPARATO: l'agente non può leggerlo nemmeno via SQL."""
        return self.data_dir / "processed" / "app.db"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.data_dir / "processed" / "checkpoints.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "processed" / "chroma"

    @property
    def index_manifest_path(self) -> Path:
        """Impronte dei manuali indicizzati: serve alla reindicizzazione incrementale."""
        return self.data_dir / "processed" / "index_manifest.json"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def model_path(self) -> Path:
        """Il modello in produzione (il "campione" promosso nel registry)."""
        return self.artifacts_dir / "failure_model.joblib"

    @property
    def model_card_path(self) -> Path:
        return self.artifacts_dir / "model_card.json"

    @property
    def reference_profile_path(self) -> Path:
        """Distribuzioni dei dati di training: il riferimento per il drift."""
        return self.artifacts_dir / "reference_profile.json"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def charts_dir(self) -> Path:
        return self.reports_dir / "charts"

    @property
    def trace_path(self) -> Path:
        return self.logs_dir / "traces.jsonl"

    @property
    def app_log_path(self) -> Path:
        return self.logs_dir / "app.log"

    @property
    def infra_dir(self) -> Path:
        return PROJECT_ROOT / "infra"

    @property
    def llm_spec(self) -> str:
        """Identificativo leggibile del modello in uso, es. "ollama:qwen3:8b"."""
        return f"{self.llm_provider}:{self.llm_model}"

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.llm_spec

    @property
    def uses_default_password(self) -> bool:
        return self.admin_password.get_secret_value() == DEFAULT_ADMIN_PASSWORD

    def ensure_dirs(self) -> None:
        for path in (self.data_dir / "raw", self.data_dir / "processed", self.artifacts_dir,
                     self.models_dir, self.figures_dir, self.charts_dir, self.logs_dir,
                     self.backups_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Restituisce sempre la stessa istanza (lru_cache = calcolata una volta sola)."""
    # Le librerie esterne (Langfuse, OpenAI, Anthropic) leggono le loro chiavi
    # dalle variabili d'ambiente: load_dotenv copia il .env in os.environ
    # senza sovrascrivere ciò che è già impostato nel terminale.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = Settings()
    settings.ensure_dirs()
    return settings
