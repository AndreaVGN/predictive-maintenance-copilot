# =============================================================================
# Makefile — scorciatoie per i comandi più usati. Esegui "make help".
# (Su macOS e Linux make è già installato.)
#
# Tutti i comandi usano gli eseguibili dentro .venv/bin: funzionano anche se
# ti dimentichi di attivare l'ambiente virtuale con "source .venv/bin/activate".
#
# Primo avvio:   make venv install models setup
# =============================================================================

# Python con cui creare l'ambiente virtuale. Serve una versione tra 3.11 e 3.14:
# per le versioni più nuove (es. 3.15) molte librerie non hanno ancora i pacchetti
# precompilati (wheel) e pip prova a compilarle, fallendo.
# Per usarne un'altra: make venv PYTHON=python3.13
PYTHON ?= python3.12
VENV   := .venv
BIN    := $(VENV)/bin

LANGFUSE_DIR := infra/langfuse
LANGFUSE_COMPOSE_URL := https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml
LANGFUSE_COMPOSE := docker compose -p pmc-langfuse -f $(LANGFUSE_DIR)/docker-compose.yml

.PHONY: help venv install models setup eda train score ingest chat api ui test lint smoke eval eval-fast eval-gate simulate drift backup pdf pdf-deps docs docker langfuse-up langfuse-down langfuse-logs clean

help:            ## mostra questo aiuto
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:            ## crea l'ambiente virtuale .venv con Python 3.12 (PYTHON=... per cambiarlo)
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
		printf "\n'%s' non trovato. Su Mac installalo con:  brew install python@3.12\n\n" "$(PYTHON)"; exit 1; }
	@$(PYTHON) -c "import sys; v=sys.version_info[:2]; sys.exit(0 if (3,11) <= v <= (3,14) else 1)" || { \
		printf "\nServe Python 3.11-3.14 (trovato: %s). Esempio: make venv PYTHON=python3.12\n\n" "$$($(PYTHON) --version)"; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	@printf "\nAmbiente creato con %s. Ora: make install\n" "$$($(BIN)/python --version)"

install:         ## installa il pacchetto e gli strumenti di sviluppo in .venv
	@test -x $(BIN)/python || { printf "\nAmbiente virtuale mancante: esegui prima  make venv\n\n"; exit 1; }
	$(BIN)/pip install -e ".[dev,observability,cloud]"

models:          ## scarica i modelli Ollama (LLM + embedding)
	ollama pull qwen3:8b
	ollama pull nomic-embed-text

setup:           ## pipeline completa: dati -> EDA -> modello -> scoring -> indice RAG
	$(BIN)/pmc --plain-logs setup

eda:             ## solo analisi esplorativa
	$(BIN)/pmc --plain-logs eda

train:           ## solo addestramento del modello
	$(BIN)/pmc --plain-logs train

score:           ## solo scoring batch
	$(BIN)/pmc --plain-logs score

ingest:          ## solo indicizzazione dei manuali
	$(BIN)/pmc --plain-logs ingest

chat:            ## chat con l'agente nel terminale
	$(BIN)/pmc chat

api:             ## avvia API e console web su http://localhost:8000 (API docs su /docs)
	$(BIN)/uvicorn pmcopilot.api.main:app --reload --port 8000

ui: api          ## alias di "api": la console web è servita dalla stessa API

test:            ## test automatici (non servono Ollama né internet)
	$(BIN)/pytest

lint:            ## controllo stile del codice
	$(BIN)/ruff check src tests tools

smoke:           ## smoke test contro l'API avviata (in un altro terminale: make api)
	PMC_SMOKE_URL=http://localhost:8000 $(BIN)/pytest -m smoke tests/smoke

eval:            ## valutazione completa (serve Ollama)
	$(BIN)/pmc --plain-logs eval all

eval-fast:       ## solo valutazioni senza LLM
	$(BIN)/pmc --plain-logs eval guardrails

eval-gate:       ## quality gate sugli ultimi run (come in CI): esce con errore se non passa
	$(BIN)/pmc --plain-logs eval gate --kind guardrails
	$(BIN)/pmc --plain-logs eval gate --kind retrieval
	$(BIN)/pmc --plain-logs eval gate --kind model

simulate:        ## 24 ore di letture simulate: make simulate SCENARIO=seasonal
	$(BIN)/pmc --plain-logs simulate --scenario $(or $(SCENARIO),normal) --hours $(or $(HOURS),24)
	$(BIN)/pmc --plain-logs score
	$(BIN)/pmc --plain-logs drift check

drift:           ## controllo del drift sulle ultime 24 ore
	$(BIN)/pmc --plain-logs drift check

backup:          ## copia di database, modello e profilo in backups/
	$(BIN)/pmc --plain-logs backup

pdf:             ## rigenera la guida in PDF (docs/guida/guida.pdf; la prima volta: make pdf-deps)
	$(BIN)/python tools/build_pdf.py

pdf-deps:        ## dipendenze per rigenerare la guida (Markdown, Chromium per la stampa)
	$(BIN)/pip install -e ".[pdf]"
	$(BIN)/playwright install chromium

docs:            ## documentazione in locale su http://localhost:8001
	$(BIN)/mkdocs serve -a localhost:8001

docker:          ## API in Docker (Ollama resta sul Mac)
	docker compose up --build

langfuse-up:     ## avvia Langfuse in Docker, già configurato (http://localhost:3000)
	@command -v docker >/dev/null 2>&1 || { printf "\nDocker non trovato: installa Docker Desktop e riprova.\n\n"; exit 1; }
	$(BIN)/pmc langfuse-init
	@test -f $(LANGFUSE_DIR)/docker-compose.yml || curl -fsSL $(LANGFUSE_COMPOSE_URL) -o $(LANGFUSE_DIR)/docker-compose.yml
	$(LANGFUSE_COMPOSE) up -d
	@printf "\nLangfuse si sta avviando: apri http://localhost:3000 tra 2-3 minuti.\nRiavvia l'API (Ctrl+C e make api) per collegarla.\n\n"

langfuse-down:   ## ferma Langfuse (i dati restano nei volumi Docker)
	$(LANGFUSE_COMPOSE) down

langfuse-logs:   ## log del server web di Langfuse
	$(LANGFUSE_COMPOSE) logs -f langfuse-web

clean:           ## cancella dati generati, modello, report e log (non l'ambiente virtuale)
	rm -rf data/raw/*.csv data/raw/SOURCE.txt data/processed artifacts reports logs
