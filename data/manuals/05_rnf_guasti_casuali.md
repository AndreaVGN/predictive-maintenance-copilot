---
doc_type: guasto
failure_mode: "RNF"
revision: 2026-10
---

# Guasto RNF — guasti casuali

## Descrizione del guasto

Il guasto RNF (Random Failure) raggruppa i guasti che non dipendono dalle variabili
monitorate. Ha una probabilità di circa 0,1% per lavorazione, cioè circa un guasto ogni
mille lavorazioni. Non è prevedibile dal modello di machine learning, perché per
definizione non lascia tracce nei dati di processo.

## Cause possibili

- difetti del materiale del pezzo grezzo;
- guasti elettrici improvvisi o micro-interruzioni di alimentazione;
- errori di carico del pezzo da parte dell'operatore;
- corpi estranei nella zona di lavoro.

## Checklist di ispezione generale

1. Mettere la macchina in sicurezza secondo la procedura LOTO.
2. Ispezione visiva della zona di lavoro, del mandrino e dell'utensile.
3. Verifica degli allarmi registrati dal controllo numerico.
4. Controllo delle connessioni elettriche e dei sensori.
5. Verifica del pezzo e del lotto di materiale.
6. Registrazione dell'evento nel registro manutenzioni con una descrizione dettagliata.

Tempo di intervento tipico: da 1 a 10 ore, molto variabile.

## Cosa aspettarsi dal sistema predittivo

Un guasto RNF non è un errore del modello. Se i guasti RNF di una macchina aumentano nel
tempo, la causa non è più casuale: va aperta un'analisi dedicata con il responsabile
qualità.
