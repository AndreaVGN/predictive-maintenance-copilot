"""
style.py — uno stile grafico unico per tutti i grafici matplotlib del progetto.

PERCHÉ
    Grafici coerenti (stessi colori, stessi font, griglie discrete) sembrano un
    prodotto unico invece che una collezione di screenshot. Definiamo i colori
    per RUOLO ("normale", "guasto") e non per nome ("blu"), così il significato
    resta lo stesso in tutti i grafici: il blu è SEMPRE "nessun guasto",
    l'arancio è SEMPRE "guasto".

    Blu e arancio sono stati scelti perché restano distinguibili anche per chi
    ha una forma di daltonismo (cosa che rosso/verde non garantisce).

MATPLOTLIB IN BREVE
    * `plt.subplots()` crea una Figure (il foglio) e uno o più Axes (i grafici).
    * Si disegna SEMPRE sugli Axes: ax.plot, ax.bar, ax.scatter...
    * `fig.savefig(path)` salva; `plt.close(fig)` libera la memoria (importante
      in un server che genera grafici su richiesta).
    * Il backend "Agg" disegna su file senza aprire finestre: necessario su
      server, container e test.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # nessuna finestra grafica: solo file PNG

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Colori per ruolo
COLOR_NORMAL = "#2a78d6"  # blu: nessun guasto / serie principale
COLOR_FAILURE = "#eb6834"  # arancio: guasto / serie di confronto
COLOR_ACCENT = "#1baf7a"  # acqua: terza serie (es. soglia, baseline)
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_GRID = "#e4e3df"
COLOR_SURFACE = "#fcfcfb"

# Mappa sequenziale (un solo colore, da chiaro a scuro) per le intensità,
# es. la matrice di confusione.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("pmc_seq", ["#eef4fc", COLOR_NORMAL, "#0d3a73"])
# Mappa divergente (due colori + grigio neutro al centro) per valori con segno,
# es. le correlazioni da -1 a +1.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "pmc_div", [COLOR_FAILURE, "#f0efec", COLOR_NORMAL]
)


def apply_style() -> None:
    """Imposta i parametri globali di matplotlib (rcParams)."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_SURFACE,
            "axes.facecolor": COLOR_SURFACE,
            "savefig.facecolor": COLOR_SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": COLOR_TEXT_SECONDARY,
            "axes.edgecolor": COLOR_GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.6,
            "xtick.color": COLOR_TEXT_SECONDARY,
            "ytick.color": COLOR_TEXT_SECONDARY,
            "text.color": COLOR_TEXT,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )
