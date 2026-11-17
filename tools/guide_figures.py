"""
guide_figures.py — genera le figure "concettuali" della guida PDF.

    python tools/guide_figures.py
    -> docs/guida/img/concepts/*.png  e  docs/guida/img/report/*.png

Serve un ambiente già preparato (make setup): legge dati e modello dalle
cartelle configurate (PMC_DATA_DIR, PMC_ARTIFACTS_DIR...). Per il grafico degli
scenari di drift lavora su una COPIA temporanea del database, così i dati veri
non vengono toccati.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from pmcopilot import domain
from pmcopilot.analysis.style import (
    COLOR_ACCENT,
    COLOR_FAILURE,
    COLOR_NORMAL,
    COLOR_TEXT_SECONDARY,
    SEQUENTIAL_CMAP,
    apply_style,
)
from pmcopilot.config import get_settings
from pmcopilot.data.features import add_engineered_features
from pmcopilot.ml.drift import DRIFT_FEATURES, check_drift, psi
from pmcopilot.ml.predictor import FailurePredictor
from pmcopilot.ml.score import score_recent_readings
from pmcopilot.ml.simulate import simulate_readings
from pmcopilot.ml.train import temporal_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guida" / "img" / "concepts"
REPORT_OUT = ROOT / "docs" / "guida" / "img" / "report"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=170)
    plt.close(fig)
    print("  ", name)


def load_frame(settings) -> pd.DataFrame:
    with sqlite3.connect(settings.db_path) as conn:
        df = pd.read_sql_query("SELECT r.*, m.machine_type FROM sensor_readings r JOIN machines m USING(machine_id) "
                               "ORDER BY r.ts", conn)
    return add_engineered_features(df)


# ------------------------------------------------------------ dati e ML
def fig_imbalance(df: pd.DataFrame) -> None:
    counts = df["machine_failure"].value_counts().sort_index()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    a.bar(["nessun guasto", "guasto"], counts.values, color=[COLOR_NORMAL, COLOR_FAILURE])
    for i, v in enumerate(counts.values):
        a.text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom")
    a.set_ylim(0, counts.max() * 1.15)
    a.set_title("Classi molto sbilanciate")
    acc_dumb = counts[0] / counts.sum()
    b.barh(["Modello \"dice sempre\nnessun guasto\"", "Un buon modello"], [acc_dumb, 0.985],
           color=[COLOR_TEXT_SECONDARY, COLOR_NORMAL])
    b.set_xlim(0.9, 1.0)
    b.set_title("Accuracy: entrambi sembrano ottimi")
    b.text(acc_dumb, 0, f"  {acc_dumb:.1%}\n  ma trova 0 guasti", va="center")
    b.text(0.985, 1, "  98,5%", va="center")
    save(fig, "imbalance")


def fig_split(df: pd.DataFrame) -> None:
    n = 60
    rng = np.random.default_rng(1)
    rand = np.zeros(n)
    rand[rng.choice(n, 12, replace=False)] = 1
    temp = np.zeros(n)
    temp[-12:] = 1
    fig, axes = plt.subplots(2, 1, figsize=(9, 2.4), sharex=True)
    for ax, mask, title in [(axes[0], rand, "Split casuale: il test è sparso nel tempo (il modello \"vede il futuro\")"),
                            (axes[1], temp, "Split temporale: si addestra sul passato, si testa sul futuro")]:
        ax.bar(range(n), np.ones(n), color=[COLOR_FAILURE if m else COLOR_NORMAL for m in mask], width=0.9)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=10)
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
    axes[1].set_xlabel("tempo (letture in ordine cronologico)   ■ blu = training   ■ arancio = test")
    axes[1].set_xticks([])
    save(fig, "split")


def fig_cv() -> None:
    fig, ax = plt.subplots(figsize=(9, 2.5))
    k = 5
    for fold in range(k):
        for part in range(k):
            ax.add_patch(plt.Rectangle((part, k - fold - 1), 0.95, 0.8,
                                       color=COLOR_FAILURE if part == fold else COLOR_NORMAL))
        ax.text(-0.15, k - fold - 0.6, f"giro {fold + 1}", ha="right", va="center")
    ax.set_xlim(-1, k)
    ax.set_ylim(-0.2, k)
    ax.axis("off")
    ax.set_title("Cross-validation a 5 fold: ogni parte fa da validazione una volta (arancio)", loc="left", fontsize=10)
    save(fig, "cv")


def fig_physics(df: pd.DataFrame) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    sample = df.sample(min(4000, len(df)), random_state=0)
    ok, ko = sample[sample.pwf == 0], sample[sample.pwf == 1]
    a.scatter(ok.rpm, ok.torque_nm, s=4, alpha=.35, color=COLOR_NORMAL, label="nessun guasto PWF")
    a.scatter(ko.rpm, ko.torque_nm, s=14, color=COLOR_FAILURE, label="guasto PWF")
    rpm = np.linspace(1150, 2900, 200)
    for p in (domain.PWF_MIN_POWER_W, domain.PWF_MAX_POWER_W):
        a.plot(rpm, p / (rpm * 2 * np.pi / 60), color=COLOR_ACCENT, lw=2)
        a.text(rpm[-1], p / (rpm[-1] * 2 * np.pi / 60), f" {p:.0f} W", color=COLOR_ACCENT, va="center")
    a.set_xlabel("giri [rpm]")
    a.set_ylabel("coppia [Nm]")
    a.set_ylim(0, 80)
    a.set_title("Coppia e giri: il confine è una CURVA", fontsize=10)
    a.legend(loc="upper right", fontsize=8)
    b.hist([df.loc[df.pwf == 0, "power_w"], df.loc[df.pwf == 1, "power_w"]], bins=60, stacked=True,
           color=[COLOR_NORMAL, COLOR_FAILURE], label=["nessun guasto", "guasto PWF"])
    b.set_yscale("log")
    for p in (domain.PWF_MIN_POWER_W, domain.PWF_MAX_POWER_W):
        b.axvline(p, color=COLOR_ACCENT, lw=2)
    b.set_xlabel("potenza = coppia × ω [W]")
    b.set_title("Con la potenza basta un taglio (due soglie)", fontsize=10)
    b.legend(fontsize=8)
    save(fig, "physics")


def fig_threshold(settings, df: pd.DataFrame) -> None:
    predictor = FailurePredictor(settings.model_path)
    _, test_idx = temporal_split(df)
    test = df.loc[test_idx]
    proba = predictor.predict_frame(test)["failure_proba"].to_numpy()
    y = test["machine_failure"].to_numpy()
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(thr, prec[:-1], color=COLOR_NORMAL, label="precision (quanti allarmi sono veri)")
    ax.plot(thr, rec[:-1], color=COLOR_FAILURE, label="recall (quanti guasti trovo)")
    ax.plot(thr, f1[:-1], color=COLOR_ACCENT, label="F1", lw=2.2)
    ax.axvline(predictor.threshold, color="#333", ls="--", lw=1)
    ax.text(predictor.threshold, 0.04, f"  soglia scelta {predictor.threshold:.2f}", fontsize=9)
    ax.axvline(0.5, color=COLOR_TEXT_SECONDARY, ls=":", lw=1)
    ax.text(0.5, 0.12, "  0,5 (default)", fontsize=9, color=COLOR_TEXT_SECONDARY)
    ax.set_xlabel("soglia sulla probabilità")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Alzare la soglia: meno falsi allarmi, ma più guasti persi", loc="left", fontsize=10)
    save(fig, "threshold")


def fig_hdf_concept(df: pd.DataFrame) -> None:
    sample = df.sample(min(5000, len(df)), random_state=2)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ok, ko = sample[sample.hdf == 0], sample[sample.hdf == 1]
    ax.scatter(ok.rpm, ok.temp_diff_k, s=4, alpha=.3, color=COLOR_NORMAL, label="nessun guasto HDF")
    ax.scatter(ko.rpm, ko.temp_diff_k, s=16, color=COLOR_FAILURE, label="guasto HDF")
    ax.add_patch(plt.Rectangle((1100, 7), domain.HDF_MAX_RPM - 1100, domain.HDF_MAX_TEMP_DIFF_K - 7,
                               fill=False, ec=COLOR_FAILURE, lw=2))
    ax.add_patch(plt.Rectangle((1100, 7), domain.HDF_MAX_RPM - 1100, domain.HDF_MAX_TEMP_DIFF_K + 1.4 - 7,
                               fill=False, ec="#6b4fa8", lw=2, ls="--"))
    box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2}
    ax.text(1400, 12.4, "linea tratteggiata: dopo il concept drift il guasto arriva già con ΔT < 10 K",
            color="#6b4fa8", fontsize=9, bbox=box)
    ax.text(1400, 12.0, "linea continua: regola appresa dal modello, ΔT < 8,6 K e giri < 1380",
            color=COLOR_FAILURE, fontsize=9, bbox=box)
    ax.set_xlim(1100, 2200)
    ax.set_ylim(7, 13)
    ax.set_xlabel("giri [rpm]")
    ax.set_ylabel("ΔT processo − aria [K]")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Guasto per dissipazione termica (HDF) e concept drift", loc="left", fontsize=10)
    save(fig, "hdf_concept")


# --------------------------------------------------------------- drift
def fig_psi_example(df: pd.DataFrame) -> None:
    # Dati illustrativi (continui): temperatura "di training" ~ N(300 K, 2 K) e
    # "di oggi" spostata di +1 K, cioè mezza deviazione standard.
    rng = np.random.default_rng(0)
    ref = rng.normal(300, 2, 10_000)
    now = rng.normal(301, 2, 3_000)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)[1:-1]))
    e = np.bincount(np.digitize(ref, edges), minlength=len(edges) + 1) / len(ref)
    a = np.bincount(np.digitize(now, edges), minlength=len(edges) + 1) / len(now)
    contrib = (np.clip(a, 1e-4, None) - np.clip(e, 1e-4, None)) * np.log(np.clip(a, 1e-4, None) / np.clip(e, 1e-4, None))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(e))
    left.bar(x - 0.2, e, 0.4, color=COLOR_NORMAL, label="training (atteso)")
    left.bar(x + 0.2, a, 0.4, color=COLOR_FAILURE, label="ultime 24 h (osservato)")
    left.set_xticks(x, [f"B{i + 1}" for i in x], fontsize=8)
    left.set_ylabel("quota di letture nel bin")
    left.set_title("10 bin = decili del training (10% ciascuno)", fontsize=10)
    left.legend(fontsize=8)
    right.bar(x, contrib, color=COLOR_ACCENT)
    right.set_xticks(x, [f"B{i + 1}" for i in x], fontsize=8)
    right.set_title(f"Contributo di ogni bin: PSI totale = {psi(e, a):.2f}", fontsize=10)
    save(fig, "psi_example")


def fig_drift_scenarios(settings) -> None:
    """Esegue gli scenari su una copia del database e disegna PSI e calo di prestazioni."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        s = settings.model_copy(update={"data_dir": tmp / "data"})
        (tmp / "data" / "processed").mkdir(parents=True)
        shutil.copy2(settings.db_path, s.db_path)
        rows, labels = [], []
        for scenario, intensity in [("normal", 1), ("seasonal", 1), ("sensor_bias", 1),
                                    ("late_replacement", 2), ("concept", 1.5)]:
            simulate_readings(s, hours=24, scenario=scenario, intensity=intensity, seed=5)
            score_recent_readings(s)
            report = check_drift(s, hours=24, save=False)
            rows.append([report["features"][f]["psi"] for f in DRIFT_FEATURES]
                        + [report["prediction"]["psi"], max(report["performance"].get("max_drop", 0), 0)])
            labels.append(scenario)
    data = np.array(rows)
    cols = [*DRIFT_FEATURES, "PSI previsioni", "calo PR-AUC/recall"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.imshow(np.minimum(data, 1.0), cmap=SEQUENTIAL_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else "#1e262d")
    ax.set_title("Cosa vede il monitoraggio in ogni scenario (PSI: >0,1 attenzione, >0,25 allarme)",
                 loc="left", fontsize=10)
    ax.grid(False)
    save(fig, "drift_scenarios")


# ------------------------------------------------------------- RAG e LLM
def fig_cosine() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    vecs = {"\"il mandrino si surriscalda\"": (0.9, 0.35), "\"temperatura alta del mandrino\"": (0.8, 0.5),
            "\"come si apre un ordine di lavoro\"": (0.15, 0.95)}
    colors = [COLOR_NORMAL, COLOR_ACCENT, COLOR_FAILURE]
    for (label, (x, y)), c in zip(vecs.items(), colors, strict=True):
        ax.annotate("", (x, y), (0, 0), arrowprops={"arrowstyle": "-|>", "color": c, "lw": 2.2})
        ax.text(x + 0.02, y + 0.02, label, color=c, fontsize=9)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.1)
    ax.set_aspect("equal")
    ax.set_title("Embedding: frasi simili = vettori vicini (angolo piccolo)", loc="left", fontsize=10)
    ax.set_xlabel("dimensione 1 (in realtà sono 768)")
    ax.set_ylabel("dimensione 2")
    save(fig, "cosine")


def fig_bm25() -> None:
    tf = np.arange(0, 16)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for k1, c in [(0.5, COLOR_ACCENT), (1.5, COLOR_NORMAL), (3.0, COLOR_FAILURE)]:
        a.plot(tf, tf * (k1 + 1) / (tf + k1), marker="o", ms=3, color=c, label=f"k1 = {k1}")
    a.plot(tf, tf, color=COLOR_TEXT_SECONDARY, ls=":", label="conteggio puro")
    a.set_ylim(0, 6)
    a.set_xlabel("quante volte la parola compare nel chunk (tf)")
    a.set_title("Saturazione: la 10ª ripetizione vale poco", fontsize=10)
    a.legend(fontsize=8)
    n = np.array([1, 2, 5, 10, 50, 200])
    idf = np.log((200 - n + 0.5) / (n + 0.5) + 1)
    b.bar([str(x) for x in n], idf, color=COLOR_NORMAL)
    b.set_xlabel("in quanti chunk (su 200) compare la parola")
    b.set_title("IDF: le parole rare pesano di più", fontsize=10)
    save(fig, "bm25")


def fig_context() -> None:
    parts = [("istruzioni\n+ schema DB", 1500, COLOR_NORMAL), ("conversazione\nprecedente", 2400, COLOR_ACCENT),
             ("risultati\ndei tool", 2600, COLOR_FAILURE), ("risposta", 700, "#6b4fa8")]
    fig, ax = plt.subplots(figsize=(9, 2.2))
    left = 0
    for label, n, c in parts:
        ax.barh(0, n, left=left, color=c)
        ax.text(left + n / 2, 0, f"{label}\n{n}", ha="center", va="center", color="white", fontsize=8.5)
        left += n
    ax.barh(0, 8192 - left, left=left, color="#e4e3df")
    ax.text(left + (8192 - left) / 2, 0, f"libero\n{8192 - left}", ha="center", va="center", fontsize=8)
    ax.axvline(8192 * 0.7, color="#333", ls="--", lw=1)
    ax.text(8192 * 0.7, 0.45, "70%: parte il riassunto", fontsize=8, ha="center")
    ax.grid(False)
    ax.set_xlim(0, 8192)
    ax.set_yticks([])
    ax.set_xlabel("finestra di contesto: num_ctx = 8192 token")
    save(fig, "context")


def copy_report_figures(settings) -> None:
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    for f in (settings.reports_dir / "figures").glob("*.png"):
        shutil.copy2(f, REPORT_OUT / f.name)
        print("   report/", f.name)


def main() -> None:
    settings = get_settings()
    apply_style()
    df = load_frame(settings)
    print(f"Figure della guida ({len(df)} letture)")
    fig_imbalance(df)
    fig_split(df)
    fig_cv()
    fig_physics(df)
    fig_threshold(settings, df)
    fig_hdf_concept(df)
    fig_psi_example(df)
    fig_drift_scenarios(settings)
    fig_cosine()
    fig_bm25()
    fig_context()
    copy_report_figures(settings)


if __name__ == "__main__":
    main()
