# 11. Produzione

Questa pagina riassume le parti aggiunte nella versione 0.2, quella "da produzione".

## Cosa c'è di nuovo

| Area | Cosa | Dove nel codice |
|---|---|---|
| Accesso | utenti admin e operatore, cookie firmato, Basic auth, blocco dopo 5 tentativi | `api/auth.py` |
| Limiti | rate limit per utente, budget giornaliero di token, token massimi per richiesta | `api/auth.py`, `agent/middleware.py` |
| LLM | Ollama o API cloud (OpenAI, Anthropic), timeout, retry, fallback, costi | `agent/llm.py` |
| Agente | middleware LangChain 1.x, riassunto della storia, checkpointer SQLite | `agent/graph.py` |
| RAG | ricerca ibrida (embedding + BM25 + RRF), filtri sui metadati, reindicizzazione incrementale | `rag/` |
| Guardrail | classificatore LLM di injection, injection indiretta, redazione dei dati personali | `agent/guardrails.py`, `agent/middleware.py` |
| Modello | registry, champion/challenger, promozione e rollback | `ml/registry.py`, `ml/train.py` |
| Drift | PSI, KS, prediction drift, concept drift, cinque scenari simulati | `ml/drift.py`, `ml/simulate.py` |
| Operazioni | scheduler, alert con webhook, backup consistenti, audit | `ops/`, `observability/alerts.py` |
| Valutazione | storico dei run, confronto A/B caso per caso, quality gate in CI | `evaluation/compare.py`, `store.py` |
| Console | pagine Esperimenti e Monitoraggio per l'amministratore | `api/static/` |

## Comandi principali

```bash
make api                          # console (admin / admin)
make simulate SCENARIO=seasonal   # dati simulati con drift
make eval && make eval-gate       # valutazione e quality gate
make backup                       # copia di database e modello
make smoke                        # smoke test contro l'API avviata
```
