# 8. Observability

**File:** `observability/structured_logging.py`, `observability/tracing.py`, `observability/metrics.py`

## Tre pilastri

| Pilastro | Domanda | Qui |
|---|---|---|
| **Log** | cosa è successo? | JSON su stderr, con `request_id` |
| **Trace** | come è andata *questa* richiesta, passo per passo? | `logs/traces.jsonl` + Langfuse |
| **Metriche** | come va il sistema *in generale*? | `GET /metrics` |

## Log strutturati

```json
{"ts": "2026-11-02T10:31:02.114+00:00", "level": "INFO", "logger": "pmcopilot.agent.service",
 "request_id": "a1b2c3d4e5f6", "message": "chat completata", "route": "in_scope",
 "latency_ms": 4210.3, "llm_calls": 3, "tools": {"sql_query": 1}, "guard_flags": []}
```

Ogni riga è un oggetto JSON e si analizza con pandas: `pd.read_json("api.log", lines=True)`.

## Tracing con i callback di LangChain

LangChain e LangGraph chiamano i metodi di un `BaseCallbackHandler` a ogni evento:

```text
on_chain_start(node=input_guard) ... on_chain_end
on_chain_start(node=agent)
  on_chat_model_start ... on_llm_end (token)
  on_tool_start(sql_query) ... on_tool_end
  on_chat_model_start ... on_llm_end
on_chain_end(agent)
```

`JsonlTraceHandler` misura le durate e conta chiamate, token e errori dei tool. Il riepilogo finisce nella risposta dell'API (`trace`). Per esempio:

```json
{"latency_ms": 4210, "llm_calls": 3, "tokens_in": 2840, "tokens_out": 210,
 "tool_calls": {"sql_query": 1, "search_manual": 1}, "tool_errors": 0,
 "node_ms": {"input_guard": 0.3, "router": 610.2, "agent": 3580.9, "output_guard": 0.2}}
```

!!! example "Analizzare le trace con pandas"
    ```python
    import pandas as pd
    t = pd.read_json("logs/traces.jsonl", lines=True)
    t[t.event == "tool_end"].groupby("name")["duration_ms"].describe()
    ```

## Langfuse (consigliato per il portfolio)

Langfuse è una piattaforma open source di LLM observability. Mostra ogni trace come un albero navigabile: per ogni chiamata al modello vedi il prompt esatto, la risposta, i tool scelti con i loro argomenti, i token e le latenze. Gira in locale con Docker.

### Avvio con un comando

Serve Docker Desktop avviato. Poi:

```bash
make langfuse-up
```

Il comando fa tre cose:

1. `pmc langfuse-init` genera segreti casuali in `infra/langfuse/.env` e scrive le stesse chiavi nel `.env` del progetto, attivando `PMC_LANGFUSE_ENABLED=true`. Stampa anche email e password per entrare.
2. Scarica il `docker-compose.yml` ufficiale di Langfuse in `infra/langfuse/` (solo la prima volta).
3. Avvia i servizi con `docker compose`: web, worker, Postgres, ClickHouse, Redis e MinIO.

Dopo 2-3 minuti apri <http://localhost:3000> ed entra con le credenziali stampate (le ritrovi in `infra/langfuse/.env`, alle righe `LANGFUSE_INIT_USER_*`). Organizzazione, progetto e chiavi esistono già: è la **headless initialization** di Langfuse, che al primo avvio legge le variabili `LANGFUSE_INIT_*` e crea tutto da sola.

Riavvia l'API (Ctrl+C, poi `make api`) perché legga il nuovo `.env`. Da quel momento:

- ogni risposta nella console ha il link **Apri la trace su Langfuse**;
- l'intestazione del pannello Esecuzione mostra il link **trace** della richiesta in corso;
- la vista Stato mostra Langfuse tra i componenti.

```bash
make langfuse-logs
make langfuse-down
```

`make langfuse-logs` mostra i log del server web. `make langfuse-down` ferma tutto; i dati restano nei volumi Docker.

### Come è collegato al codice

`tracing.build_callbacks` genera un id di trace (`Langfuse.create_trace_id()`) e lo passa al `CallbackHandler` di Langfuse, che si affianca al nostro `JsonlTraceHandler`. Conoscere l'id in anticipo permette di costruire il link prima che la richiesta finisca.

`tracing.langfuse_available` controlla una volta se Langfuse risponde (`auth_check`). Se il server è spento, l'app continua con il solo tracing su file e ricontrolla ogni minuto. Non ci sono errori a ogni richiesta.

!!! warning "Porte occupate"
    Il compose di Langfuse usa le porte 3000 (web), 9090 (MinIO) e, solo in locale, 5432 (Postgres), 6379 (Redis), 8123 e 9000 (ClickHouse). Se hai già un Postgres o un Redis sul Mac, fermali o cambia le porte in `infra/langfuse/docker-compose.yml`.

### Cosa guardare in una trace

Apri la trace di *"Quanti guasti HDF ha avuto la M02 e cosa raccomanda il manuale?"*. Nell'albero trovi:

1. i nodi `input_guard` e `router` (con il prompt del router e il JSON strutturato restituito);
2. il nodo `agent`, con dentro il ciclo `model → tools → model`;
3. le chiamate a `sql_query` e `search_manual`, con argomenti e output;
4. il nodo `output_guard`.

Confronta i tempi: quasi tutta la latenza sta nelle chiamate al modello, non nei tool.

## Metriche

`GET /metrics` restituisce: richieste totali, errori, distribuzione delle route, guardrail scattati, uso dei tool, token, latenza p50/p95. In produzione si esporrebbero in formato Prometheus e si visualizzerebbero in Grafana.

## Esercizi

1. Calcola da `traces.jsonl` quale tool ha la latenza p95 più alta.
2. Aggiungi al `JsonlTraceHandler` il conteggio dei **retry**, cioè di quante volte lo stesso tool viene richiamato dopo un `ERRORE`.
3. Esponi `/metrics` in formato Prometheus con `prometheus_client`.
4. Con Langfuse attivo, invia come *score* il voto del giudice della valutazione (`langfuse.create_score`) e confronta le versioni del prompt.
5. Aggiungi un allarme: se la latenza p95 supera 15 s o il tasso di errore supera il 5%, scrivi un log `WARNING`.
