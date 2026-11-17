# Per iniziare

## Requisiti

- **Python tra 3.11 e 3.14** (consigliato 3.12). Su Mac: `brew install python@3.12`
- [Ollama](https://ollama.com) installato e avviato (su Mac: l'app nella barra dei menu)
- circa 6 GB liberi per i modelli
- Docker solo se vuoi Langfuse o l'API in container (opzionale)

!!! warning "Perché non l'ultima versione di Python?"
    Le librerie con parti compilate (pydantic-core, numpy, onnxruntime per Chroma...) pubblicano i pacchetti precompilati (*wheel*) per una nuova versione di Python solo qualche mese dopo il suo rilascio. Senza wheel, pip prova a compilare dai sorgenti e di solito fallisce con errori come `resolution-too-deep`. Per questo `make venv` usa Python 3.12 e rifiuta le versioni non supportate.

## Installazione

Un comando per riga, da copiare così come sono:

```bash
git clone https://github.com/<tuo-utente>/predictive-maintenance-copilot.git
cd predictive-maintenance-copilot
make venv
make install
cp .env.example .env
make models
```

| Comando | Cosa fa |
|---|---|
| `make venv` | crea l'ambiente virtuale `.venv` con Python 3.12 (per un'altra versione: `make venv PYTHON=python3.13`) |
| `make install` | `pip install -e ".[dev,observability]"` dentro `.venv` |
| `cp .env.example .env` | crea la configurazione (modello, URL di Ollama...) |
| `make models` | `ollama pull qwen3:8b` e `ollama pull nomic-embed-text` |

Tutti i comandi `make` usano gli strumenti dentro `.venv`, quindi non serve attivare l'ambiente. Per usare `pmc`, `pytest` o `jupyter` direttamente, attivalo con `source .venv/bin/activate`.

!!! note "Commenti nel terminale (zsh)"
    In zsh, il terminale di default su Mac, il carattere `#` incollato in una riga di comando **non** è un commento: viene passato come argomento (`make: No rule to make target '#'`). Per questo i blocchi di comandi di questa guida non contengono commenti.

!!! note "Quale modello?"
    Serve un modello con **tool calling**. Il default è `qwen3:8b`. Se usi già un altro modello (per esempio `gemma4`), imposta `PMC_LLM_MODEL` nel `.env`. Con modelli piccoli (3-4B) l'agente funziona ma sbaglia più spesso la scelta dei tool: la [valutazione](09-evaluation.md) te lo mostrerà in numeri.

## Costruire tutto

```bash
make setup
```

Esegue cinque passi, che poi studierai uno per uno:

| Passo | Comando singolo | Cosa produce |
|---|---|---|
| 1. Dati | `pmc data` | `data/raw/ai4i2020.csv`, `data/processed/plant.db` |
| 2. EDA | `pmc eda` | grafici in `reports/figures/eda_*.png`, `reports/eda_summary.md` |
| 3. Training | `pmc train` | `artifacts/failure_model.joblib`, `reports/model_report.md` |
| 4. Scoring | `pmc score` | tabella `predictions` nel database |
| 5. Indice RAG | `pmc ingest` | `data/processed/chroma/` |

Il passo 1 prova a scaricare il dataset AI4I 2020 da UCI. Se non ci riesce, genera dati sintetici equivalenti e lo scrive in `data/raw/SOURCE.txt`.

## Usarlo

=== "Terminale"
    ```bash
    make chat
    ```
    ```text
    tu > quali macchine sono a rischio alto?
    copilota > ...
      [tool: sql_query · 4210 ms · 2 chiamate LLM]
    ```

=== "Console web"
    ```bash
    make api
    ```
    Apri <http://localhost:8000>: chat con esecuzione in diretta, previsione con gli slider, impianto, valutazioni e stato. Vedi [API e console](07-api.md#la-console-web).

=== "API"
    ```bash
    make api
    ```
    Apri <http://localhost:8000/docs>: trovi tutti gli endpoint, provabili dal browser.
    ```bash
    curl -u admin:admin -X POST localhost:8000/chat -H "Content-Type: application/json" \
         -d '{"message": "Cosa dice il manuale sul guasto HDF?"}'
    ```

=== "Notebook"
    ```bash
    .venv/bin/pip install jupyterlab
    .venv/bin/jupyter lab notebooks/01_eda.ipynb
    ```

## Verificare che funzioni

```bash
make test
make eval
```

`make test` esegue circa 50 test e non ha bisogno di Ollama; `make eval` fa la valutazione completa con Ollama e richiede qualche minuto.

## Problemi comuni

| Sintomo | Causa probabile | Soluzione |
|---|---|---|
| `Connection refused` su 11434 | Ollama non è avviato | apri l'app Ollama o `ollama serve` |
| `model "qwen3:8b" not found` | modello non scaricato | `ollama pull qwen3:8b` |
| `Database non trovato` | setup non eseguito | `make setup` |
| L'agente risponde senza usare i tool | modello troppo piccolo | prova un modello più grande o abbassa la temperatura |
| Risposte lente | modello thinking | `PMC_LLM_REASONING=false` nel `.env` |
