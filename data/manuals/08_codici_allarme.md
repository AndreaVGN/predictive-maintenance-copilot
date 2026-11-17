---
doc_type: codici
failure_mode: "*"
revision: 2026-10
---

# Codici di allarme del controllo numerico PMX

> Documento fittizio a scopo didattico. I codici sono inventati ma coerenti con le
> modalità di guasto degli altri manuali.

## Come leggere un codice

Ogni allarme ha un codice nel formato AL-XXX. La prima cifra indica la famiglia:
1 = utensile, 2 = temperatura, 3 = azionamento e potenza, 4 = carico meccanico,
5 = sistema e sicurezza. Il controllo numerico registra codice, data e valore misurato.

## Famiglia 1: utensile

| Codice | Significato | Modalità collegata | Azione |
|---|---|---|---|
| AL-101 | Usura utensile oltre 180 minuti | TWF | programmare la sostituzione |
| AL-102 | Usura utensile oltre 200 minuti | TWF | sostituire l'utensile entro il turno |
| AL-108 | Rottura utensile rilevata | TWF | fermo immediato, sostituzione e pezzo di prova |

## Famiglia 2: temperatura

| Codice | Significato | Modalità collegata | Azione |
|---|---|---|---|
| AL-201 | Delta termico sotto 9,5 K a bassa velocità | HDF | verificare ventola e refrigerante |
| AL-207 | Sovratemperatura mandrino | HDF | fermare la macchina, pulire lo scambiatore |
| AL-209 | Sonda di temperatura aria guasta | nessuna | sostituire la sonda: i dati di temperatura non sono affidabili |

## Famiglia 3: azionamento e potenza

| Codice | Significato | Modalità collegata | Azione |
|---|---|---|---|
| AL-301 | Potenza sotto 3500 W | PWF | verificare i parametri di taglio |
| AL-302 | Potenza sopra 9000 W | PWF | ridurre avanzamento o profondità di passata |
| AL-315 | Allarme inverter del mandrino | PWF | intervento del tecnico elettrico abilitato |

## Famiglia 4: carico meccanico

| Codice | Significato | Modalità collegata | Azione |
|---|---|---|---|
| AL-401 | Usura per coppia oltre il 90% del limite | OSF | ridurre il carico di taglio |
| AL-404 | Sovraccarico mandrino | OSF | fermo, verifica serraggio pezzo e utensile |

## Famiglia 5: sistema e sicurezza

| Codice | Significato | Modalità collegata | Azione |
|---|---|---|---|
| AL-501 | Porta di protezione aperta in ciclo | nessuna | il ciclo si ferma: chiudere la porta e riarmare |
| AL-512 | Micro-interruzione di alimentazione | RNF | ispezione generale secondo checklist |
| AL-520 | Pulsante di emergenza premuto | nessuna | verificare la causa prima del riarmo, sempre con LOTO |

## Allarmi ripetuti

Lo stesso codice ripetuto più di tre volte nello stesso turno richiede un ordine di lavoro
con priorità P2, anche se la macchina continua a funzionare.
