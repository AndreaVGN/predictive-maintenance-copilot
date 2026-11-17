"""
domain.py — la "conoscenza di dominio" del progetto in un solo posto.

PERCHÉ ESISTE
    Le stesse soglie fisiche compaiono in quattro punti diversi:
      * il generatore di dati sintetici (data/synthetic.py) le usa per decidere
        quando avviene un guasto;
      * il predittore (ml/predictor.py) le usa per spiegare i "fattori di rischio";
      * i manuali di manutenzione (data/manuals/*.md) le riportano come limiti
        operativi, e il RAG li recupera;
      * i test e la valutazione le usano come risposte attese.
    Tenerle qui evita incoerenze: se cambi una soglia, la cambi una volta sola.

LE CINQUE MODALITÀ DI GUASTO (dal dataset AI4I 2020)
    TWF  Tool Wear Failure        usura utensile tra 200 e 240 minuti
    HDF  Heat Dissipation Failure delta termico < 8.6 K e giri < 1380 rpm
    PWF  Power Failure            potenza < 3500 W oppure > 9000 W
    OSF  Overstrain Failure       usura x coppia oltre una soglia per tipo
    RNF  Random Failure           0.1% di probabilità, nessuna causa misurabile
"""

from __future__ import annotations

# Tipi di prodotto/macchina: L = low, M = medium, H = high quality.
MACHINE_TYPES: tuple[str, ...] = ("L", "M", "H")

# Minuti di usura che ogni lavorazione aggiunge all'utensile, per tipo.
TOOL_WEAR_INCREMENT: dict[str, int] = {"L": 2, "M": 3, "H": 5}

# TWF: l'utensile viene sostituito o si rompe in questa finestra (minuti).
TOOL_WEAR_REPLACEMENT_MIN: int = 200
TOOL_WEAR_REPLACEMENT_MAX: int = 240

# HDF: smaltimento del calore insufficiente.
HDF_MAX_TEMP_DIFF_K: float = 8.6
HDF_MAX_RPM: int = 1380

# PWF: finestra di potenza meccanica ammessa (watt).
PWF_MIN_POWER_W: float = 3500.0
PWF_MAX_POWER_W: float = 9000.0

# OSF: limite del prodotto usura [min] x coppia [Nm], per tipo di macchina.
OSF_LIMIT: dict[str, float] = {"L": 11_000.0, "M": 12_000.0, "H": 13_000.0}

# RNF: probabilità di guasto casuale per lavorazione.
RNF_PROBABILITY: float = 0.001

FAILURE_MODES: tuple[str, ...] = ("TWF", "HDF", "PWF", "OSF", "RNF")

FAILURE_MODE_NAMES: dict[str, str] = {
    "TWF": "usura utensile",
    "HDF": "dissipazione termica",
    "PWF": "potenza fuori range",
    "OSF": "sovraccarico (overstrain)",
    "RNF": "guasto casuale",
}

# Margine con cui segnalare che un valore si sta "avvicinando" a una soglia.
WARNING_MARGIN: float = 0.10  # 10%
