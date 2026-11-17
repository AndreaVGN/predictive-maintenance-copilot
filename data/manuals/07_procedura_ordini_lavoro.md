---
doc_type: procedura
failure_mode: "*"
revision: 2026-10
---

# Procedura per gli ordini di lavoro di manutenzione

## Quando aprire un ordine di lavoro

Si apre un ordine di lavoro quando:

- il sistema di previsione indica rischio alto per una macchina;
- un operatore rileva un sintomo descritto nei documenti sulle modalità di guasto;
- un parametro è vicino a una soglia (entro il 10%) per più lavorazioni consecutive;
- dopo un guasto, per pianificare l'intervento correttivo.

## Livelli di priorità

| Priorità | Quando | Tempo massimo di presa in carico |
|---|---|---|
| P1 | guasto in corso o rischio alto con parametro già oltre soglia | 1 ora |
| P2 | rischio alto senza parametri oltre soglia, oppure rischio medio persistente | 8 ore (entro il turno) |
| P3 | manutenzione preventiva e segnalazioni a bassa urgenza | 5 giorni lavorativi |

## Contenuto dell'ordine di lavoro

Ogni ordine di lavoro deve indicare: codice macchina, modalità di guasto sospetta
(TWF, HDF, PWF, OSF, RNF o generico), priorità, una nota descrittiva con i valori
osservati e la fonte della segnalazione (operatore, sistema predittivo, ispezione).

## Approvazione

Gli ordini proposti dall'assistente digitale devono essere approvati da una persona
(responsabile di linea o manutenzione) prima di essere registrati. L'assistente non può
aprire ordini in autonomia.

## Chiusura

L'ordine si chiude quando l'intervento è registrato nel registro manutenzioni con azione
eseguita, tempo di fermo e nome del tecnico.
