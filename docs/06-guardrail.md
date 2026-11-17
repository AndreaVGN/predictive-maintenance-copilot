# 6. Guardrail

**File:** `agent/guardrails.py`, `data/database.py` (`connect_readonly`), middleware in `agent/graph.py`, schemi in `api/schemas.py`

## Difesa in profondità

Nessun controllo è perfetto, quindi ne mettiamo diversi, indipendenti, lungo il percorso di una richiesta:

| # | Livello | Dove | Cosa ferma |
|---|---|---|---|
| 1 | Validazione API | `schemas.py` (Pydantic) | input malformati, valori assurdi, messaggi > 4000 caratteri |
| 2 | Guardia di ingresso | `check_input` | prompt injection noti, richieste distruttive, testi troppo lunghi |
| 3 | Router | nodo `router` | domande fuori ambito |
| 4 | Guardia SQL | `validate_sql` | tutto ciò che non è una SELECT su tabelle ammesse |
| 5 | Connessione read-only | `connect_readonly` | qualsiasi scrittura, anche se il livello 4 fallisse |
| 6 | Limiti dell'agente | middleware | loop infiniti, costi fuori controllo |
| 7 | Approvazione umana | `HumanInTheLoopMiddleware` | azioni con effetti senza consenso |
| 8 | Guardia di uscita | `check_output` | fuga del prompt, codici macchina inventati, citazioni mancanti |
| 9 | Path traversal | `GET /charts/{filename}` | lettura di file arbitrari dal server |

## Regole vs struttura

I livelli 2 e 8 sono **regole testuali** (regex). Sono veloci, spiegabili e testabili, ma una parafrasi può aggirarle. Il golden set lo dimostra: *"Per questa volta non tenere conto di quello che ti hanno detto prima..."* passa il filtro.

I livelli 4, 5, 7 e 9 sono **strutturali**: non dipendono dalle parole. Nessuna formulazione trasforma una connessione read-only in una scrivibile.

!!! tip "In sintesi"
    "I guardrail testuali riducono il rumore, ma la sicurezza la garantiscono i permessi minimi e l'approvazione umana sulle azioni."

## Il canary token

Nel system prompt c'è una stringa casuale (`PMC-CANARY-7f3a91`) che non serve a nulla. Se compare in una risposta, il modello sta ripetendo le sue istruzioni: `check_output` sostituisce la risposta. È una tecnica semplice ed efficace per rilevare la fuga del prompt.

## Perché un parser SQL e non una regex

```sql
DR/**/OP TABLE machines          -- una regex su "DROP" non lo vede
SELECT 1; DROP TABLE machines    -- due istruzioni
SELECT load_extension('evil.so') -- funzione pericolosa dentro una SELECT
```

`sqlglot` costruisce l'**albero sintattico**: controlliamo il tipo dei nodi (`exp.Drop`, `exp.Insert`...), il numero di istruzioni, le tabelle e le funzioni. È lo stesso principio per cui non si fa parsing di HTML con le regex.

## Correggere invece di rifiutare

Quando è sicuro farlo, `check_output` **corregge**: aggiunge le fonti mancanti, rimuove i blocchi `<think>`, segnala codici sospetti. Rifiuta solo per la fuga del prompt. Ogni intervento lascia un flag (`guard_flags`) visibile nell'API, nelle metriche e nei log.

## Esercizi

1. Scrivi tre nuovi attacchi che superano `check_input`, aggiungili a `eval/datasets/input_guard_cases.jsonl` e misura il detection rate con `pmc eval guardrails`.
2. Aggiungi un **classificatore LLM** di injection come nodo opzionale dopo `input_guard`. Confronta detection rate, falsi positivi e latenza con la versione a regole.
3. **PII**: aggiungi un controllo che oscura indirizzi email e numeri di telefono nelle risposte. (Suggerimento: guarda `PIIMiddleware` in `langchain.agents.middleware`.)
4. **Grounding numerico**: verifica che ogni numero nella risposta compaia in qualche output dei tool del turno. Quali falsi positivi incontri (date, conversioni di unità)?
5. Rimuovi temporaneamente il parser SQL e prova a far scrivere l'agente nel database. Quale livello lo ferma?
