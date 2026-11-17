# 9. Evaluation

**File:** `evaluation/*.py`, `eval/datasets/*.jsonl`, `ml/train.py`

## Perché valutare

"Mi sembra che funzioni" non è una metrica. Senza valutazione non puoi rispondere a domande come: *il nuovo prompt è meglio del vecchio?*, *conviene passare da qwen3:8b a un modello più piccolo?*, *il chunk size giusto è 500 o 900?*

## Quattro livelli

| Livello | Comando | Metriche | Serve Ollama |
|---|---|---|---|
| Modello ML | `pmc train` | PR-AUC, ROC-AUC, precision, recall, F1, matrice di confusione | no |
| Guardrail | `pmc eval guardrails` | detection rate, false positive rate | no |
| Retrieval | `pmc eval retrieval` | hit@1, hit@4, MRR | sì (embedding) |
| Agente | `pmc eval agent` | route accuracy, tool selection, keyword hit, faithfulness 1-5, latenza | sì |

`pmc eval all` le esegue tutte e produce `reports/eval_report.md`.

## I golden set

Sono file JSONL scritti a mano, in `eval/datasets/`:

```json
{"id": "a06", "question": "In quali condizioni si verifica un guasto per dissipazione termica?",
 "expected_route": "in_scope", "expected_tools": ["search_manual"], "must_contain": [["8,6", "8.6"], "1380"]}
```

`must_contain` accetta alternative: `["8,6", "8.6"]` significa che basta una delle due.

## Metriche di retrieval

- **hit@k**: la fonte giusta è tra i primi k risultati?
- **MRR** (Mean Reciprocal Rank): media di `1/posizione`. Vale 1 se la fonte giusta è prima, 0,5 se seconda, 0 se assente.

## LLM-as-a-judge

Un secondo LLM legge domanda, output dei tool e risposta, e assegna un voto di **fedeltà** (*faithfulness*) da 1 a 5 con output strutturato (`JudgeVerdict`).

| Pro | Contro |
|---|---|
| scala a centinaia di casi | anche il giudice sbaglia |
| vede la fedeltà, che le regole non vedono | tende a premiare risposte lunghe |
| economico rispetto a un umano | un giudice piccolo è poco affidabile |

Per questo lo affianchiamo a metriche deterministiche (tool usati, parole chiave) e conviene usare come giudice un modello più grande (`PMC_JUDGE_MODEL`).

## Leggere i risultati con onestà

- Il golden set dei guardrail è scritto insieme alle regole: un 100% sarebbe ottimista. Per questo contiene due attacchi parafrasati che passano il filtro (detection < 100%).
- Il modello ML su AI4I ha metriche molto alte perché i guasti seguono regole deterministiche. Su dati reali le prestazioni sarebbero più basse.
- 16 domande per l'agente sono poche per conclusioni statistiche: servono a confrontare versioni, non a certificare il sistema.

## Esercizi

1. **Regressione**: salva i risultati di `pmc eval agent`, cambia il system prompt, rivaluta e confronta. Scrivi uno script che segnala le domande peggiorate.
2. **Confronto modelli**: esegui la valutazione dell'agente con due modelli Ollama diversi. Costruisci una tabella qualità/latenza.
3. **Calibrare il giudice**: dai tu un voto a mano a 10 risposte e calcola l'accordo con il giudice (correlazione o kappa di Cohen).
4. **Ragas**: prova la libreria `ragas` per le metriche RAG (context precision, answer relevancy) con Ollama come backend.
5. **CI**: aggiungi una soglia minima (per esempio tool selection ≥ 0,8) che fa fallire la pipeline se la qualità scende.
