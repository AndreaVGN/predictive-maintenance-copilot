---
doc_type: guasto
failure_mode: "OSF"
revision: 2026-10
---

# Guasto OSF — sovraccarico (overstrain)

## Descrizione del guasto

Il guasto OSF (Overstrain Failure) si verifica quando l'utensile, già usurato, lavora con
una coppia elevata. La grandezza da controllare è il prodotto tra usura dell'utensile
(minuti) e coppia (Nm). Il guasto avviene quando questo prodotto supera un limite che
dipende dal tipo di macchina:

| Tipo macchina | Limite usura x coppia |
|---|---|
| L | 11000 min·Nm |
| M | 12000 min·Nm |
| H | 13000 min·Nm |

Esempio: su una macchina di tipo L, un utensile con 200 minuti di usura che lavora a
60 Nm raggiunge 12000 min·Nm, oltre il limite di 11000.

## Sintomi e segnali premonitori

- prodotto usura x coppia oltre il 90% del limite del proprio tipo;
- coppia in crescita su un utensile già usurato;
- scheggiature sul tagliente o deformazione del pezzo.

## Interventi correttivi

1. Ridurre il carico di taglio (profondità di passata o avanzamento) per abbassare la coppia.
2. Se l'usura è alta, anticipare la sostituzione dell'utensile (vedi documento TWF).
3. Verificare il serraggio del pezzo: un pezzo non ben bloccato aumenta la coppia.

Tempo di intervento tipico: da 2 a 5 ore.

## Differenza tra OSF e TWF

TWF dipende solo dall'usura (l'utensile arriva a fine vita). OSF dipende dalla combinazione
di usura e carico: un utensile a metà vita può andare in OSF se lavora con coppie molto alte.
