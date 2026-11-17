---
doc_type: guasto
failure_mode: "PWF"
revision: 2026-10
---

# Guasto PWF — potenza fuori range

## Descrizione del guasto

Il guasto PWF (Power Failure) si verifica quando la potenza meccanica richiesta dalla
lavorazione esce dalla finestra ammessa dall'azionamento: sotto 3500 W oppure sopra
9000 W. La potenza si calcola come coppia per velocità angolare:
potenza [W] = coppia [Nm] x rpm x 2π / 60.

Esempio: con coppia 40 Nm e velocità 1500 rpm la potenza è circa 6283 W, quindi nella
norma. Con coppia 15 Nm e velocità 2000 rpm la potenza scende a circa 3142 W, sotto il
minimo.

## Cause tipiche

- potenza troppo bassa: velocità molto alta con coppia molto bassa, tipica di passate
  di finitura con parametri non corretti o di utensili che perdono presa sul pezzo;
- potenza troppo alta: coppia elevata a velocità ancora alta, tipica di passate di
  sgrossatura troppo aggressive o di materiale più duro del previsto.

## Diagnosi

1. Leggere coppia e velocità delle ultime lavorazioni e calcolare la potenza.
2. Verificare che il programma CNC usi i parametri di taglio approvati per quel prodotto.
3. Controllare l'azionamento del mandrino (allarmi dell'inverter, connessioni).
4. Verificare il materiale del pezzo grezzo rispetto alla specifica.

## Interventi correttivi

- riportare i parametri di taglio nei valori approvati (avanzamento, profondità di passata);
- taratura dei parametri dell'azionamento da parte del tecnico elettrico;
- se la potenza resta vicina ai limiti (entro il 10% da 3500 W o da 9000 W), segnalare la
  macchina al responsabile di linea anche in assenza di guasto.

Tempo di intervento tipico: da 3 a 8 ore, perché spesso richiede il tecnico elettrico.
