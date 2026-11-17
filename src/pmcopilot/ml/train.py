"""
train.py — addestramento e valutazione del modello di previsione guasti.

IL PROBLEMA
    Classificazione binaria: data una lavorazione (temperature, giri, coppia,
    usura, tipo macchina) stimare la probabilità che porti a un guasto.
    Difficoltà principale: i guasti sono RARI (~3-5% delle righe). Un modello
    che dice sempre "nessun guasto" ha il 96% di accuracy ed è inutile.
    Per questo:
      * usiamo uno split STRATIFICATO (stessa % di guasti in train e test);
      * pesiamo di più la classe rara (class_weight="balanced");
      * misuriamo con PR-AUC (average precision), non con l'accuracy;
      * scegliamo la SOGLIA di decisione invece di usare 0.5 per default.

IL FLUSSO DI QUESTO FILE
    1. carica le letture dal database SQLite (SQL -> pandas)
    2. aggiunge le feature fisiche (numpy)
    3. divide in train (80%) e test (20%), stratificando
    4. confronta 4 modelli x 2 set di feature con cross-validation sul train
    5. sceglie il migliore e calibra la soglia (sempre solo sul train!)
    6. valuta UNA volta sul test: è la stima onesta delle prestazioni
    7. salva modello + "model card" (JSON) + grafici + report Markdown

    Regola d'oro: il test set si guarda una sola volta, alla fine. Se lo usi
    per scegliere modello o soglia, la stima diventa ottimista (overfitting
    sulla valutazione).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pmcopilot.config import Settings
from pmcopilot.data.database import load_readings
from pmcopilot.data.features import (
    CATEGORICAL_FEATURES,
    TARGET,
    add_engineered_features,
    feature_columns,
)
from pmcopilot.ml import plots
from pmcopilot.ml.drift import build_reference_profile
from pmcopilot.ml.registry import promote, version_dir
from pmcopilot.store import get_store

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- costruzione
def candidate_models(seed: int, quick: bool = False) -> dict[str, object]:
    """I modelli da confrontare. `quick=True` (nei test) ne usa meno e più leggeri."""
    models: dict[str, object] = {
        # Baseline: predice sempre la frequenza dei guasti. Serve come "pavimento":
        # se un modello non la batte nettamente, non sta imparando nulla.
        "baseline (dummy)": DummyClassifier(strategy="prior"),
        # Modello lineare: semplice, interpretabile, veloce.
        "logistic regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        # Gradient boosting su istogrammi: di solito il più forte su dati tabellari.
        "hist gradient boosting": HistGradientBoostingClassifier(
            class_weight="balanced", learning_rate=0.08, max_iter=300, random_state=seed
        ),
    }
    if not quick:
        # Random forest: molti alberi su campioni diversi, robusta e poco sensibile ai parametri.
        models["random forest"] = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    return models


def build_pipeline(model: object, use_engineered: bool) -> Pipeline:
    """Pipeline = preprocessing + modello in un unico oggetto.

    Perché è importante: lo scaler impara media e deviazione standard SOLO dai
    dati di training di ogni fold. Se scalassimo tutto il dataset prima dello
    split, informazioni del test "trapelerebbero" nel training (data leakage).
    La Pipeline lo impedisce per costruzione, ed è anche ciò che salviamo su
    disco: in produzione basta chiamare pipeline.predict_proba(dati_grezzi).
    """
    numeric = [c for c in feature_columns(use_engineered) if c not in CATEGORICAL_FEATURES]
    preprocessing = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", clone(model))])


# ------------------------------------------------------------------ utilities
def choose_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Sceglie la soglia che massimizza F1 sulle predizioni out-of-fold del train.

    precision_recall_curve prova tutte le soglie possibili; F1 è la media
    armonica di precision e recall, quindi premia un buon equilibrio tra
    "trovare i guasti" e "non dare falsi allarmi".
    In un caso reale la soglia si sceglierebbe con i costi: quanto costa un
    fermo non previsto vs un'ispezione inutile? (vedi esercizi in docs/).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # l'ultimo punto della curva non ha soglia
    return float(thresholds[best]), float(f1[best])


@dataclass
class TrainingResult:
    model_name: str
    model_version: str
    threshold: float
    test_metrics: dict[str, float]
    comparison: pd.DataFrame
    champion_metrics: dict[str, float] | None = None  # il modello in produzione, sugli stessi dati
    promoted: bool = False


def temporal_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.Index, pd.Index]:
    """Train = il passato, test = le letture più RECENTI.

    È lo split onesto per un sistema che dovrà predire il FUTURO: con uno split
    casuale il modello vede nel training letture successive a quelle del test
    (ad es. la stessa macchina un'ora dopo) e la stima risulta ottimista.
    """
    ordered = df.sort_values("ts").index
    cut = int(len(ordered) * (1 - test_size))
    return ordered[:cut], ordered[cut:]


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate_test": float(y_true.mean()),
        "n_test": int(len(y_true)),
    }


# ---------------------------------------------------------------- entry point
def train_and_evaluate(settings: Settings, quick: bool = False, split: str = "temporal",
                       promote_policy: str = "if_better", user: str = "system",
                       label: str | None = None) -> TrainingResult:
    """Addestra, valuta, registra e (se migliore) promuove un nuovo modello.

    promote_policy: "if_better" (champion/challenger), "always" o "never".
    """
    settings.ensure_dirs()
    seed = settings.random_seed

    # 1-2. Dati dal database + feature fisiche
    df = add_engineered_features(load_readings(settings.db_path))
    y = df[TARGET].to_numpy()
    logger.info("Righe: %d, guasti: %d (%.2f%%)", len(df), y.sum(), 100 * y.mean())

    # 3. Split 80/20: temporale (default, realistico) o casuale stratificato
    all_features = feature_columns(use_engineered=True)
    if split == "temporal":
        train_idx, test_idx = temporal_split(df)
        X_train, X_test = df.loc[train_idx, all_features], df.loc[test_idx, all_features]
        y_train, y_test = df.loc[train_idx, TARGET].to_numpy(), df.loc[test_idx, TARGET].to_numpy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df[all_features], y, test_size=0.2, stratify=y, random_state=seed
        )

    # 4. Confronto modelli x set di feature, con cross-validation sul solo train
    cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=seed)
    rows = []
    for model_name, model in candidate_models(seed, quick).items():
        for feature_set, use_eng in (("grezze", False), ("fisiche", True)):
            pipe = build_pipeline(model, use_eng)
            scores = cross_validate(
                pipe,
                X_train[feature_columns(use_eng)],
                y_train,
                cv=cv,
                scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
                n_jobs=1,
            )
            rows.append(
                {
                    "model": model_name,
                    "features": feature_set,
                    "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
                    "cv_pr_auc_std": scores["test_pr_auc"].std(),
                    "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                }
            )
            logger.info("CV %-24s feature %-8s PR-AUC %.3f", model_name, feature_set,
                        scores["test_pr_auc"].mean())
    comparison = pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False)

    # 5. Il migliore (le feature fisiche vincono quasi sempre: lo verifichiamo, non lo assumiamo)
    best = comparison.iloc[0]
    best_name, use_eng = str(best["model"]), best["features"] == "fisiche"
    features = feature_columns(use_eng)
    best_pipe = build_pipeline(candidate_models(seed, quick)[best_name], use_eng)

    # Soglia calibrata su predizioni "out-of-fold": ogni riga del train è
    # predetta da un modello che NON l'ha vista in addestramento.
    oof_proba = cross_val_predict(best_pipe, X_train[features], y_train, cv=cv,
                                  method="predict_proba")[:, 1]
    threshold, oof_f1 = choose_threshold(y_train, oof_proba)
    logger.info("Migliore: %s (feature %s), soglia %.3f (F1 OOF %.3f)",
                best_name, best["features"], threshold, oof_f1)

    # 6. Addestramento finale su tutto il train e valutazione unica sul test
    best_pipe.fit(X_train[features], y_train)
    proba_test = best_pipe.predict_proba(X_test[features])[:, 1]
    pred_test = (proba_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred_test)
    test_metrics = _metrics(y_test, proba_test, threshold)

    # 6b. Champion/challenger: il modello in produzione sugli STESSI dati di test.
    store = get_store(settings.app_db_path)
    champion_metrics = None
    if settings.model_path.exists():
        try:
            champion = joblib.load(settings.model_path)
            champ_proba = champion["pipeline"].predict_proba(X_test[champion["features"]])[:, 1]
            champion_metrics = _metrics(y_test, champ_proba, champion["threshold"])
            champion_metrics["version"] = champion["version"]
        except Exception as exc:  # modello vecchio incompatibile: niente confronto
            logger.warning("Champion non valutabile: %s", exc)

    # 7a. Salvataggio nel registry: una cartella per versione
    slug = best_name.split(" (")[0].replace(" ", "_")  # es. "random_forest"
    version = f"{slug}-{datetime.now(UTC):%Y%m%d%H%M%S}"
    vdir = version_dir(settings, version)
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "threshold": threshold, "features": features, "version": version},
                vdir / "model.joblib")
    source = settings.data_source_path.read_text().strip() if settings.data_source_path.exists() else "unknown"
    model_card = {
        "model_version": version,
        "algorithm": best_name,
        "feature_set": str(best["features"]),
        "features": features,
        "threshold": threshold,
        "split": split,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_source": source,
        "train_period": [str(df.loc[X_train.index, "ts"].min()), str(df.loc[X_train.index, "ts"].max())],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_metrics": test_metrics,
        "champion_on_same_test": champion_metrics,
        "intended_use": "Supporto decisionale alla manutenzione, non sostituisce le procedure di sicurezza.",
        "limitations": "Addestrato su dati sintetici AI4I 2020: le prestazioni su macchine reali vanno verificate.",
    }
    (vdir / "model_card.json").write_text(json.dumps(model_card, indent=2, ensure_ascii=False))
    # Profilo di riferimento per il drift: feature del training e probabilità del
    # modello finale sul TEST (dati mai visti). Non le probabilità sul training:
    # un random forest sui propri dati dà valori quasi tutti 0 o 1 (li ha
    # "memorizzati") e il confronto con i dati nuovi sarebbe falsato.
    train_frame = df.loc[X_train.index]
    profile = build_reference_profile(train_frame, proba_test, threshold, seed, test_metrics)
    (vdir / "reference_profile.json").write_text(json.dumps(profile))

    store.register_model(version=version, algorithm=best_name, feature_set=str(best["features"]),
                         split=split, metrics={**test_metrics, "champion": champion_metrics},
                         data_source=source, path=str(vdir), status="candidate")
    store.audit(user, "train_model", version, split=split, pr_auc=test_metrics["pr_auc"])

    # 7a-bis. Promozione: se non c'è un campione, o se lo sfidante è migliore.
    better = champion_metrics is None or test_metrics["pr_auc"] >= champion_metrics["pr_auc"]
    promoted = promote_policy == "always" or (promote_policy == "if_better" and better)
    if promoted:
        promote(settings, version, user)
    logger.info("Modello %s: PR-AUC %.3f (campione: %s) -> %s", version, test_metrics["pr_auc"],
                f"{champion_metrics['pr_auc']:.3f}" if champion_metrics else "nessuno",
                "PROMOSSO" if promoted else "resta candidato")

    # Storico per la pagina Esperimenti (tipo "model")
    store.save_eval_run(
        "model", {"pr_auc": test_metrics["pr_auc"], "roc_auc": test_metrics["roc_auc"],
                  "precision": test_metrics["precision"], "recall": test_metrics["recall"],
                  "f1": test_metrics["f1"], "threshold": threshold, "promoted": promoted,
                  "champion_pr_auc": champion_metrics["pr_auc"] if champion_metrics else None},
        [{"id": f"{r['model']} / {r['features']}", "question": f"{r['model']} (feature {r['features']})",
          **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}}
         for r in comparison.to_dict("records")],
        config={"version": version, "algorithm": best_name, "feature_set": str(best["features"]),
                "split": split, "quick": quick, "data_source": source, "n_train": int(len(y_train))},
        label=label or version, user=user)

    # 7b. Grafici
    fig_dir = settings.figures_dir
    precision, recall, _ = precision_recall_curve(y_test, proba_test)
    plots.plot_precision_recall(recall, precision, test_metrics["pr_auc"], float(y_test.mean()),
                                fig_dir / "model_pr_curve.png",
                                chosen=(test_metrics["recall"], test_metrics["precision"]))
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    plots.plot_roc(fpr, tpr, test_metrics["roc_auc"], fig_dir / "model_roc_curve.png")
    plots.plot_confusion_matrix(cm, fig_dir / "model_confusion_matrix.png", threshold)
    plots.plot_model_comparison(comparison, fig_dir / "model_comparison.png")
    perm = permutation_importance(best_pipe, X_test[features], y_test, scoring="average_precision",
                                  n_repeats=5 if quick else 10, random_state=seed, n_jobs=1)
    plots.plot_feature_importance(pd.Series(perm.importances_mean, index=features),
                                  pd.Series(perm.importances_std, index=features),
                                  fig_dir / "model_feature_importance.png")

    # 7c. Report Markdown
    _write_report(settings, model_card, comparison, cm)
    return TrainingResult(best_name, version, threshold, test_metrics, comparison, champion_metrics, promoted)


def _write_report(settings: Settings, card: dict, comparison: pd.DataFrame, cm: np.ndarray) -> None:
    m = card["test_metrics"]
    table = comparison.to_markdown(index=False, floatfmt=".3f")
    report = f"""# Model report — {card['model_version']}

Dati: **{card['data_source']}** · split **{card['split']}** · train {card['n_train']} righe · test {card['n_test']} righe ·
guasti nel test {m['positive_rate_test']:.2%}

## Modello scelto
**{card['algorithm']}** con feature **{card['feature_set']}**, soglia di decisione **{card['threshold']:.3f}**
(scelta massimizzando F1 su predizioni out-of-fold del train).

## Metriche sul test set (valutato una sola volta)
| PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |

Matrice di confusione (righe = reale, colonne = predetto): TN {cm[0, 0]}, FP {cm[0, 1]}, FN {cm[1, 0]}, TP {cm[1, 1]}

## Confronto in cross-validation
{table}

## Grafici
![PR](figures/model_pr_curve.png)
![Confronto](figures/model_comparison.png)
![Confusione](figures/model_confusion_matrix.png)
![Importanza](figures/model_feature_importance.png)
"""
    (settings.reports_dir / "model_report.md").write_text(report)
