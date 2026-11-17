"""
eda.py — Exploratory Data Analysis con pandas, numpy e matplotlib.

PERCHÉ FARE EDA PRIMA DEL MODELLO
    Prima di addestrare qualunque cosa bisogna guardare i dati: quanti guasti
    ci sono, come si distribuiscono le variabili, quali relazioni esistono.
    Qui scopriamo, per esempio, che i guasti PWF stanno lungo due "curve" nel
    piano giri-coppia: le iso-potenze a 3500 W e 9000 W. È la prova visiva che
    la feature `power_w` sarà utile.

COSA IMPARI QUI (pandas)
    * df.describe(): statistiche di base per colonna
    * df.groupby("col")[...].agg(...): aggregazioni per gruppo (come GROUP BY in SQL)
    * df.corr(): matrice di correlazione
    * maschere booleane: df[df["machine_failure"] == 1]

Le funzioni restituiscono DataFrame o salvano figure, così il notebook
(notebooks/01_eda.ipynb) e la CLI (pmc eda) riusano lo stesso codice.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmcopilot import domain
from pmcopilot.analysis.style import (
    COLOR_FAILURE,
    COLOR_NORMAL,
    COLOR_TEXT_SECONDARY,
    DIVERGING_CMAP,
    apply_style,
)
from pmcopilot.config import Settings
from pmcopilot.data.database import load_readings
from pmcopilot.data.features import (
    ENGINEERED_FEATURES,
    FAILURE_MODE_COLUMNS,
    RAW_NUMERIC_FEATURES,
    TARGET,
    add_engineered_features,
)


# ------------------------------------------------------------------ tabelle
def load_eda_frame(settings: Settings) -> pd.DataFrame:
    """Dati dal database + feature fisiche: il punto di partenza dell'analisi."""
    return add_engineered_features(load_readings(settings.db_path))


def failure_rate_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Numero di lavorazioni, guasti e tasso di guasto per tipo di macchina."""
    return (
        df.groupby("machine_type")
        .agg(n_readings=(TARGET, "size"), n_failures=(TARGET, "sum"), failure_rate=(TARGET, "mean"))
        .reindex(list(domain.MACHINE_TYPES))
    )


def failure_mode_counts(df: pd.DataFrame) -> pd.Series:
    """Quanti guasti per modalità (una lavorazione può averne più di una)."""
    counts = df[FAILURE_MODE_COLUMNS].sum()
    counts.index = [c.upper() for c in counts.index]
    return counts.sort_values(ascending=False)


# ------------------------------------------------------------------ grafici
def plot_distributions(df: pd.DataFrame, path: Path) -> None:
    """Istogrammi delle variabili, separati per guasto / nessun guasto."""
    apply_style()
    cols = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    ok, ko = df[df[TARGET] == 0], df[df[TARGET] == 1]
    for ax, col in zip(axes.flat, cols, strict=False):
        bins = np.histogram_bin_edges(df[col], bins=40)
        # density=True: confrontiamo le FORME delle distribuzioni, non i conteggi
        # (altrimenti i pochi guasti sarebbero invisibili).
        ax.hist(ok[col], bins=bins, density=True, alpha=0.55, color=COLOR_NORMAL, label="nessun guasto")
        ax.hist(ko[col], bins=bins, density=True, alpha=0.65, color=COLOR_FAILURE, label="guasto")
        ax.set_title(col, fontsize=10)
        ax.set_yticks([])
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Distribuzioni delle variabili: guasti vs lavorazioni regolari", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_speed_torque(df: pd.DataFrame, path: Path) -> None:
    """Giri vs coppia: i guasti di potenza seguono le curve di iso-potenza."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ok, ko = df[df[TARGET] == 0], df[df[TARGET] == 1]
    ax.scatter(ok["rpm"], ok["torque_nm"], s=6, alpha=0.25, color=COLOR_NORMAL, label="nessun guasto")
    ax.scatter(ko["rpm"], ko["torque_nm"], s=14, alpha=0.9, color=COLOR_FAILURE,
               edgecolor="white", linewidth=0.4, label="guasto")
    # Iso-potenza: coppia = P * 60 / (2*pi*rpm). Calcolo vettoriale con numpy.
    rpm_grid = np.linspace(df["rpm"].min(), df["rpm"].max(), 200)
    for power in (domain.PWF_MIN_POWER_W, domain.PWF_MAX_POWER_W):
        ax.plot(rpm_grid, power * 60 / (2 * np.pi * rpm_grid), color=COLOR_TEXT_SECONDARY,
                linestyle="--", linewidth=1)
        ax.annotate(f"{power:.0f} W", xy=(rpm_grid[-1], power * 60 / (2 * np.pi * rpm_grid[-1])),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=COLOR_TEXT_SECONDARY)
    ax.set_xlabel("Velocità di rotazione [rpm]")
    ax.set_ylabel("Coppia [Nm]")
    ax.set_title("Velocità vs coppia, con i limiti di potenza")
    ax.legend(loc="upper right", markerscale=2)
    fig.savefig(path)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, path: Path) -> None:
    """Matrice di correlazione (Pearson) con mappa divergente centrata sullo zero."""
    apply_style()
    cols = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.grid(False)
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlazioni tra variabili")
    fig.savefig(path)
    plt.close(fig)


def plot_failure_modes(df: pd.DataFrame, path: Path) -> None:
    apply_style()
    counts = failure_mode_counts(df)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(counts.index, counts.to_numpy(), color=COLOR_FAILURE, width=0.6)
    ax.bar_label(bars, padding=2, fontsize=9, color=COLOR_TEXT_SECONDARY)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Numero di guasti")
    ax.set_title("Guasti per modalità")
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------- entry point
def run_eda(settings: Settings) -> dict[str, Path]:
    """Esegue l'intera EDA: salva i grafici e un riepilogo Markdown."""
    settings.ensure_dirs()
    df = load_eda_frame(settings)
    fig_dir = settings.figures_dir
    outputs = {
        "distributions": fig_dir / "eda_distributions.png",
        "speed_torque": fig_dir / "eda_speed_torque.png",
        "correlation": fig_dir / "eda_correlation.png",
        "failure_modes": fig_dir / "eda_failure_modes.png",
    }
    plot_distributions(df, outputs["distributions"])
    plot_speed_torque(df, outputs["speed_torque"])
    plot_correlation(df, outputs["correlation"])
    plot_failure_modes(df, outputs["failure_modes"])

    summary = f"""# EDA — riepilogo

Righe: {len(df)} · guasti: {int(df[TARGET].sum())} ({df[TARGET].mean():.2%})

## Tasso di guasto per tipo di macchina
{failure_rate_by_type(df).to_markdown(floatfmt=".4f")}

## Guasti per modalità
{failure_mode_counts(df).to_frame("n").to_markdown()}

## Statistiche descrittive
{df[RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES].describe().T.to_markdown(floatfmt=".2f")}

![Distribuzioni](figures/eda_distributions.png)
![Velocità vs coppia](figures/eda_speed_torque.png)
![Correlazioni](figures/eda_correlation.png)
![Modalità](figures/eda_failure_modes.png)
"""
    (settings.reports_dir / "eda_summary.md").write_text(summary)
    return outputs
