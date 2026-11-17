---
doc_type: panoramica
failure_mode: "*"
revision: 2026-10
---

# Panoramica dell'impianto e delle fresatrici serie PMX

> Documento fittizio a scopo didattico. Azienda, macchine e procedure sono inventate,
> ma le soglie corrispondono alle regole del dataset AI4I 2020.

## Struttura dello stabilimento

Lo stabilimento ha tre linee produttive di fresatrici CNC della serie PMX.
La Linea 1 lavora i prodotti di qualità L (low), la Linea 2 i prodotti di qualità M
(medium) e la Linea 3 i prodotti di qualità H (high). Ogni macchina ha un codice
nel formato M seguito da due cifre (per esempio M01, M12).

Il tipo di qualità determina quanto velocemente si consuma l'utensile: ogni lavorazione
aggiunge 2 minuti di usura sulle macchine L, 3 minuti sulle M e 5 minuti sulle H.

## Variabili monitorate

Per ogni lavorazione il sistema di acquisizione registra:

- temperatura dell'aria ambiente, in kelvin (valore tipico intorno a 300 K);
- temperatura di processo, in kelvin (tipicamente circa 10 K sopra l'aria);
- velocità di rotazione del mandrino, in giri al minuto (rpm);
- coppia, in newton metro (Nm), tipicamente intorno a 40 Nm;
- usura dell'utensile, in minuti di utilizzo dall'ultima sostituzione.

Da queste variabili si ricavano tre grandezze derivate usate per la diagnosi:
il delta termico (temperatura di processo meno temperatura dell'aria), la potenza
meccanica (coppia per velocità angolare, cioè coppia x rpm x 2π / 60) e lo sforzo
cumulato (usura utensile x coppia).

## Limiti operativi di riferimento

| Grandezza | Limite | Guasto associato |
|---|---|---|
| Usura utensile | fine vita tra 200 e 240 minuti | TWF |
| Delta termico e velocità | delta < 8,6 K insieme a velocità < 1380 rpm | HDF |
| Potenza meccanica | sotto 3500 W o sopra 9000 W | PWF |
| Usura x coppia | oltre 11000 (L), 12000 (M), 13000 (H) min·Nm | OSF |
| Guasto casuale | probabilità 0,1% per lavorazione | RNF |

## Modalità di guasto

Le cinque modalità di guasto sono descritte in documenti dedicati: usura utensile (TWF),
dissipazione termica (HDF), potenza fuori range (PWF), sovraccarico o overstrain (OSF)
e guasto casuale (RNF). Una stessa lavorazione può presentare più modalità
contemporaneamente: in quel caso si interviene partendo dalla causa più grave per la
sicurezza.

## Sistema di previsione dei guasti

Un modello di machine learning stima per ogni lavorazione la probabilità di guasto.
Il rischio è classificato come basso, medio o alto. Il rischio alto indica che la
probabilità supera la soglia di decisione del modello: in quel caso il responsabile
di linea valuta l'apertura di un ordine di lavoro secondo la procedura ordini di lavoro.
La previsione è uno strumento di supporto: non sostituisce le ispezioni né le
procedure di sicurezza.
