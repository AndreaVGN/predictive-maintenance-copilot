"""
plots.py — grafici di valutazione del modello (matplotlib).

Ogni funzione riceve i dati già calcolati e salva UN file PNG.
Separare "calcolo" (train.py) e "disegno" (qui) rende entrambi più semplici
da leggere e da testare.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmcopilot.analysis.style import (
    COLOR_ACCENT,
    COLOR_FAILURE,
    COLOR_NORMAL,
    COLOR_TEXT_SECONDARY,
    SEQUENTIAL_CMAP,
    apply_style,
)


def plot_confusion_matrix(cm: np.ndarray, path: Path, threshold: float) -> None:
    """Matrice di confusione 2x2 con conteggi e percentuali per riga."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.grid(False)
    ax.imshow(cm, cmap=SEQUENTIAL_CMAP)
    labels = ["Nessun guasto", "Guasto"]
    row_totals = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / row_totals[i, 0] if row_totals[i, 0] else 0
            # Testo bianco sulle celle scure, scuro su quelle chiare.
            color = "white" if cm[i, j] > cm.max() * 0.5 else "#0b0b0b"
            ax.text(j, i, f"{cm[i, j]}\n({pct:.0%})", ha="center", va="center", color=color)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title(f"Matrice di confusione (soglia {threshold:.2f})")
    fig.savefig(path)
    plt.close(fig)


def plot_precision_recall(
    recall: np.ndarray,
    precision: np.ndarray,
    ap: float,
    baseline: float,
    path: Path,
    chosen: tuple[float, float] | None = None,
) -> None:
    """Curva precision-recall: la metrica giusta quando i positivi sono rari."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(recall, precision, color=COLOR_NORMAL, label=f"Modello (AP = {ap:.3f})")
    # Un classificatore casuale ha precisione pari alla frazione di positivi.
    ax.axhline(baseline, color=COLOR_TEXT_SECONDARY, linestyle="--", linewidth=1,
               label=f"Casuale (AP = {baseline:.3f})")
    if chosen is not None:
        ax.scatter([chosen[0]], [chosen[1]], s=60, color=COLOR_FAILURE, zorder=3,
                   edgecolor="white", linewidth=1.5, label="Soglia scelta")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (guasti trovati)")
    ax.set_ylabel("Precision (allarmi corretti)")
    ax.set_title("Curva precision-recall (test set)")
    ax.legend(loc="center left")
    fig.savefig(path)
    plt.close(fig)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(fpr, tpr, color=COLOR_NORMAL, label=f"Modello (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=COLOR_TEXT_SECONDARY, linestyle="--", linewidth=1, label="Casuale")
    ax.set_xlabel("Tasso di falsi positivi")
    ax.set_ylabel("Tasso di veri positivi")
    ax.set_title("Curva ROC (test set)")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, path: Path) -> None:
    """Barre orizzontali: PR-AUC in cross-validation per modello e set di feature."""
    apply_style()
    pivot = results.pivot(index="model", columns="features", values="cv_pr_auc_mean")
    pivot = pivot.sort_values(by=pivot.columns[-1])
    fig, ax = plt.subplots(figsize=(6.2, 0.6 * len(pivot) + 1.4))
    y = np.arange(len(pivot))
    height = 0.38
    colors = {"grezze": COLOR_FAILURE, "fisiche": COLOR_NORMAL}
    for k, feature_set in enumerate(pivot.columns):
        values = pivot[feature_set].to_numpy()
        ax.barh(y + (k - 0.5) * height, values, height=height - 0.04,
                color=colors.get(feature_set, COLOR_ACCENT), label=f"feature {feature_set}")
        for yi, v in zip(y + (k - 0.5) * height, values, strict=True):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color=COLOR_TEXT_SECONDARY)
    ax.set_yticks(y, pivot.index)
    ax.set_xlim(0, 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PR-AUC media in cross-validation (5 fold)")
    ax.set_title("Confronto modelli: feature grezze vs feature fisiche")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, std: pd.Series, path: Path) -> None:
    """Permutation importance: quanto peggiora il modello se 'mescoliamo' una feature."""
    apply_style()
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(6.0, 0.4 * len(importances) + 1.2))
    ax.barh(importances.index, importances.to_numpy(), xerr=std[importances.index].to_numpy(),
            color=COLOR_NORMAL, error_kw={"ecolor": COLOR_TEXT_SECONDARY, "linewidth": 1})
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Calo di PR-AUC quando la feature viene permutata")
    ax.set_title("Importanza delle feature (permutation importance)")
    fig.savefig(path)
    plt.close(fig)
