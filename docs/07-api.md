# 7. API con FastAPI

**File:** `api/main.py`, `api/schemas.py`, `agent/service.py`

## Endpoint

| Metodo | Percorso | Scopo |
|---|---|---|
| GET | `/health` | stato di database, modello, indice e Ollama |
| POST | `/predict` | solo il modello ML: veloce e deterministico, senza LLM |
| POST | `/chat` | domanda all'agente (`thread_id` opzionale per continuare) |
| POST | `/chat/{thread_id}/decision` | approva o rifiuta un'azione in sospeso |
| GET | `/charts/{filename}` | immagine generata dall'agente |
| GET | `/metrics` | metriche aggregate |
| GET | `/docs` | Swagger UI generata automaticamente |
| GET | `/` | console web di test (vedi sotto) |
| POST | `/chat/stream` | come `/chat`, ma invia i passi dell'agente in tempo reale (SSE) |
| POST | `/chat/{thread_id}/decision/stream` | come `/decision`, in streaming |
| GET | `/config` | modelli in uso, origine dei dati, stato di Langfuse |
| GET | `/machines`, `/machines/{id}` | elenco macchine con rischio; storico di una macchina |
| GET | `/model` | model card e figure del training |
| POST | `/auth/login`, `/auth/logout` · GET `/auth/me` | sessione della console (cookie firmato); gli script possono usare HTTP Basic |
| GET | `/health/live`, `/health/ready` | sonde pubbliche per Docker e bilanciatori |
| POST | `/feedback` | pollice su/giù su una risposta (anche come score su Langfuse) |
| GET/POST | `/experiments/*` | **admin**: run di valutazione, confronto A/B, andamento, quality gate, nuovi run |
| GET/POST | `/ops/*` | **admin**: lavori periodici, drift, simulazione, alert, audit, registry dei modelli, training |
| GET | `/metrics/daily` (admin), `/metrics/prometheus` | utilizzo per giorno; metriche in formato Prometheus |

Tutti gli endpoint tranne `/health/live`, `/health/ready` e la pagina `/` richiedono l'autenticazione (401 altrimenti); quelli **admin** rispondono 403 agli operatori.

## La console web

Con `make api` apri <http://localhost:8000> e accedi (default `admin` / `admin`, da cambiare nel `.env`). La console ha queste viste:

| Vista | Cosa ci fai |
|---|---|
| **Copilota** | chat con l'agente; a destra il pannello **Esecuzione** mostra in diretta nodi del grafo, decisioni del modello, query SQL, argomenti e risultati dei tool, approvazioni; sotto la chat, un contatore mostra quanta finestra di contesto è occupata |
| **Previsione** | il modello ML da solo: muovi gli slider (o parti dall'ultima lettura di una macchina) e guarda probabilità, soglia, grandezze fisiche e fattori di rischio |
| **Impianto** | tutte le macchine con il rischio stimato; clic su una macchina per gli andamenti delle ultime 48 ore, gli interventi e gli ordini di lavoro |
| **Stato** | componenti attivi, configurazione, metriche dell'API e collegamenti (Swagger, Langfuse, Prometheus) |
| **Esperimenti** (admin) | lancia valutazioni, guarda l'andamento delle metriche, confronta due run caso per caso (regressioni, configurazione cambiata, diff del prompt), applica il quality gate |
| **Monitoraggio** (admin) | alert, report di drift con diagnosi, simulazione del flusso sensori, lavori periodici, registry dei modelli, utilizzo, audit |

I file sono in `src/pmcopilot/api/static/` (HTML, CSS e JavaScript senza framework). Gli endpoint dedicati alla console stanno in `api/ui_routes.py`.

## Streaming con Server-Sent Events

`/chat/stream` non aspetta la fine per rispondere. Tiene aperta la connessione e invia una riga per evento, appena succede:

```text
data: {"type": "start", "request_id": "8e5166a9da88", ...}
data: {"type": "node", "node": "input_guard", ...}
data: {"type": "node", "node": "router", "detail": {"route": "in_scope"}}
data: {"type": "model", "decided": "tool", "tokens_in": 1840, "tokens_out": 42}
data: {"type": "tool_call", "tool": "sql_query", "args": {"query": "SELECT ..."}}
data: {"type": "tool_result", "tool": "sql_query", "status": "ok", "preview": "3 righe ..."}
data: {"type": "model", "decided": "risposta", ...}
data: {"type": "node", "node": "output_guard", ...}
data: {"type": "final", "result": {...}}
```

Lato server, `CopilotService._stream` usa `graph.stream(..., stream_mode="updates", subgraphs=True)`: LangGraph restituisce l'aggiornamento dello stato dopo ogni nodo, compresi quelli del sottografo dell'agente (`model` e `tools`). Lato browser, `streamSSE` in `util.js` legge la risposta a pezzi con `fetch` e `ReadableStream`.

Puoi vederlo anche da terminale:

```bash
curl -N -u admin:admin -X POST localhost:8000/chat/stream -H "Content-Type: application/json" -d '{"message": "Quante macchine ci sono per linea?"}'
```

## Esempio completo con approvazione

1. Richiesta che richiede approvazione:

```bash
curl -s -u admin:admin -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "Apri un ordine P2 per la M05, sospetto HDF, controllare la ventola"}'
```

Risposta (estratto): `"pending_approval": [{"tool": "open_work_order", "args": {...}}], "thread_id": "abc..."`

2. Approvazione, con il `thread_id` ricevuto:

```bash
curl -s -u admin:admin -X POST localhost:8000/chat/abc.../decision -H "Content-Type: application/json" \
  -d '{"approve": true}'
```

Risposta (estratto): `"answer": "Ordine di lavoro #1 registrato..."`

Finché c'è un'approvazione in sospeso, un nuovo `/chat` sullo stesso thread restituisce **409 Conflict**: il client deve prima decidere.

## Concetti

**Lifespan.** Modello, indice e grafo si caricano una volta all'avvio, non a ogni richiesta:
```python
@asynccontextmanager
async def lifespan(app):
    app.state.service = CopilotService.from_settings(settings)
    yield
```

**App factory.** `create_app(settings, service)` permette ai test di passare un servizio con LLM finto. È *dependency injection* a livello di applicazione.

**Middleware HTTP.** Assegna un `X-Request-ID`, lo mette in una `ContextVar` letta dal logger, misura la latenza e scrive un log per richiesta.

**`def` invece di `async def`.** Il grafo è sincrono. Con `def`, FastAPI esegue l'endpoint in un thread pool e il server resta reattivo. Con `async def` e codice bloccante dentro, il server si bloccherebbe.

**Separazione dei livelli.** L'API non conosce LangGraph: chiama `CopilotService`. La stessa logica serve la CLI e la valutazione.

## Docker

Prima esegui `make setup` sul Mac (dati, modello, indice), poi:

```bash
docker compose up --build
```

Ollama resta sul Mac per usare la GPU (Metal). Il container lo raggiunge con `host.docker.internal`.

## Esercizi

1. **Streaming**: aggiungi `POST /chat/stream` che restituisce Server-Sent Events con gli aggiornamenti dei nodi (`graph.stream(..., stream_mode="updates")`).
2. **Autenticazione**: proteggi `/chat` e `/decision` con una API key in un header (`fastapi.security.APIKeyHeader`). Solo chi ha il ruolo "responsabile" può approvare.
3. **Rate limiting**: limita a 10 richieste al minuto per client. Dove lo metteresti?
4. **Endpoint batch**: `POST /predict/batch` che accetta una lista di letture e usa `predict_frame` (vettoriale). Misura la differenza di velocità rispetto a N chiamate singole.
5. Scrivi un test che verifica che `/health` restituisca `degraded` se manca l'indice Chroma.
