"""
cli.py — il comando `pmc`: tutte le operazioni del progetto da terminale.

PIPELINE
    pmc setup                     data -> eda -> train -> score -> ingest
    pmc data [--force]            scarica/genera il dataset e costruisce il database
    pmc eda                       analisi esplorativa (reports/figures)
    pmc train [--split temporal|random] [--promote if_better|always|never] [--quick]
    pmc score                     scoring batch delle letture recenti
    pmc ingest [--full]           indicizzazione (incrementale) dei manuali

USO
    pmc chat                      chat con l'agente nel terminale

VALUTAZIONE (ogni run resta nello storico di app.db)
    pmc eval guardrails [--with-llm]
    pmc eval retrieval [--mode dense|bm25|hybrid|all]
    pmc eval agent [--limit N] [--label "testo"]
    pmc eval all
    pmc eval list [--kind agent]
    pmc eval compare RUN_A RUN_B
    pmc eval gate --kind agent [--baseline RUN]     exit code 1 se non passa (per la CI)

ESERCIZIO
    pmc simulate --hours 24 --scenario seasonal [--intensity 1] [--ramp]
    pmc drift check
    pmc model list | pmc model promote VERSIONE
    pmc alerts check
    pmc backup
    pmc langfuse-init
"""

from __future__ import annotations

import argparse
import json
import sys

from pmcopilot.config import get_settings
from pmcopilot.observability.structured_logging import setup_logging_from_settings


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ------------------------------------------------------------- pipeline
def cmd_data(args) -> None:
    from pmcopilot.data.pipeline import prepare_data

    _print(prepare_data(get_settings(), force_download=args.force))


def cmd_eda(_) -> None:
    from pmcopilot.analysis.eda import run_eda

    for name, path in run_eda(get_settings()).items():
        print(f"{name:15s} {path}")


def cmd_train(args) -> None:
    from pmcopilot.ml.train import train_and_evaluate

    r = train_and_evaluate(get_settings(), quick=args.quick, split=args.split,
                           promote_policy=args.promote, user="cli")
    print(f"Modello: {r.model_name} ({r.model_version}), soglia {r.threshold:.3f}")
    print(f"PR-AUC {r.test_metrics['pr_auc']:.3f}"
          + (f" | campione attuale sugli stessi dati: {r.champion_metrics['pr_auc']:.3f}" if r.champion_metrics else "")
          + f" -> {'PROMOSSO in produzione' if r.promoted else 'resta candidato'}")


def cmd_score(_) -> None:
    from pmcopilot.ml.score import score_recent_readings

    print(f"Predizioni scritte: {score_recent_readings(get_settings())}")


def cmd_ingest(args) -> None:
    from pmcopilot.agent.llm import get_embeddings
    from pmcopilot.rag.ingest import build_vector_store

    settings = get_settings()
    _print(build_vector_store(settings, get_embeddings(settings), full=args.full))


def cmd_setup(args) -> None:
    args.full = True
    steps = [("Dati", cmd_data), ("EDA", cmd_eda), ("Training", cmd_train),
             ("Scoring", cmd_score), ("Indicizzazione manuali", cmd_ingest)]
    for i, (label, fn) in enumerate(steps, start=1):
        print(f"\n=== [{i}/{len(steps)}] {label} ===")
        fn(args)
    print("\nSetup completato. Prova: make api  (console su http://localhost:8000, utente admin)")


# ------------------------------------------------------------------- chat
def cmd_chat(_) -> None:
    from pmcopilot.agent.service import CopilotService

    settings = get_settings()
    service = CopilotService.from_settings(settings)
    thread_id = None
    print("Copilota Manutenzione — 'esci' per uscire, 'nuova' per una nuova conversazione.\n")
    while True:
        try:
            question = input("tu > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"esci", "exit", "quit"}:
            break
        if question.lower() == "nuova":
            thread_id = None
            print("(nuova conversazione)\n")
            continue
        if not question:
            continue
        result = service.chat(question, thread_id, user=settings.admin_user)
        thread_id = result.thread_id
        while result.pending_approval:
            print(f"\ncopilota > {result.answer}")
            choice = input("Approvi? [s/N] > ").strip().lower()
            result = service.decide(thread_id, approve=choice in {"s", "si", "sì", "y", "yes"},
                                    user=settings.admin_user)
        print(f"\ncopilota > {result.answer}")
        used = result.trace.get("context_peak", 0)
        meta = [f"tool: {', '.join(result.tools_used) or '-'}",
                f"{result.trace.get('latency_ms', 0):.0f} ms",
                f"{result.trace.get('llm_calls', 0)} chiamate LLM",
                f"contesto {used}/{settings.llm_num_ctx} token ({used / settings.llm_num_ctx:.0%})"]
        if result.cost_usd:
            meta.append(f"${result.cost_usd:.4f}")
        if result.charts:
            meta.append("grafici: " + ", ".join(str(settings.charts_dir / c) for c in result.charts))
        if result.guard_flags:
            meta.append(f"guardrail: {', '.join(result.guard_flags)}")
        print(f"  [{' · '.join(meta)}]\n")


# ------------------------------------------------------------- valutazione
def cmd_eval(args) -> None:
    from pmcopilot.evaluation.common import write_markdown_report
    from pmcopilot.store import get_store

    settings = get_settings()
    store = get_store(settings.app_db_path)
    if args.target == "list":
        for r in store.list_eval_runs(args.kind):
            print(f"{r['id']:38s} {r['ts']}  {r['label'] or ''}")
        return
    if args.target == "compare":
        from pmcopilot.evaluation.compare import compare_runs

        if len(args.runs) != 2:
            sys.exit("Uso: pmc eval compare RUN_A RUN_B")
        cmp = compare_runs(store, *args.runs)
        print(f"Confronto {cmp['a']['id']} -> {cmp['b']['id']}\n")
        for m in cmp["metrics"]:
            if "delta" in m:
                print(f"  {m['metric']:32s} {m['a']!s:>10} -> {m['b']!s:>10}  ({m['delta']:+.4g}, {m['verdict']})")
        print("\nConfigurazione cambiata:", cmp["config_changes"] or "nessuna")
        print("Casi:", cmp["case_counts"])
        for c in cmp["cases"]:
            if c["change"] == "regressione":
                print(f"  REGRESSIONE {c['id']}: {c['question']}")
        return
    if args.target == "gate":
        from pmcopilot.evaluation.compare import check_gate

        result = check_gate(settings, store, args.kind or "agent", args.run, args.baseline)
        for c in result["checks"]:
            print(f"  [{'OK' if c['ok'] else 'NO'}] {c['rule']}: {c['value']}")
        print("GATE SUPERATO" if result["passed"] else "GATE NON SUPERATO")
        sys.exit(0 if result["passed"] else 1)

    targets = ["guardrails", "retrieval", "agent"] if args.target == "all" else [args.target]
    for target in targets:
        print(f"\n=== Valutazione: {target} ===")
        if target == "guardrails":
            from pmcopilot.evaluation.guardrails import evaluate_guardrails

            _print(evaluate_guardrails(settings, with_llm=args.with_llm, label=args.label, user="cli"))
        elif target == "retrieval":
            from pmcopilot.evaluation.retrieval import MODES, evaluate_retrieval

            modes = MODES if args.mode == "all" else [args.mode or settings.retriever_mode]
            for mode in modes:
                _print(evaluate_retrieval(settings, mode=mode, label=args.label, user="cli"))
        else:
            from pmcopilot.evaluation.agent import evaluate_agent

            _print(evaluate_agent(settings, limit=args.limit, label=args.label, user="cli"))
    print(f"\nReport: {write_markdown_report(settings)}  ·  storico completo: console > Esperimenti")


# --------------------------------------------------------------- esercizio
def cmd_simulate(args) -> None:
    from pmcopilot.ml.simulate import simulate_readings

    _print(simulate_readings(get_settings(), args.hours, args.scenario, args.intensity, args.ramp, args.seed))


def cmd_drift(_) -> None:
    from pmcopilot.ml.drift import check_drift

    report = check_drift(get_settings())
    print(f"Stato: {report['status']}  diagnosi: {report['prediction']['diagnosis'] or 'nessuna'}")
    for name, f in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
        print(f"  {name:16s} PSI {f['psi']:.3f}  KS {f['ks_stat']:.3f}  "
              f"media {f['mean_ref']} -> {f['mean_now']}  [{f['status']}]")
    print(f"  {'previsioni':16s} PSI {report['prediction']['psi']:.3f}  [{report['prediction']['status']}]")
    _print(report["performance"])


def cmd_model(args) -> None:
    settings = get_settings()
    if args.action == "list":
        from pmcopilot.ml.registry import list_models

        for m in list_models(settings):
            print(f"{m['version']:40s} {m['status']:10s} PR-AUC {m['metrics']['pr_auc']:.3f}  {m['ts']}")
    else:
        from pmcopilot.ml.registry import promote

        promote(settings, args.version, "cli")
        print(f"{args.version} è ora in produzione (riavvia l'API o usa la console per ricaricarlo).")


def cmd_alerts(_) -> None:
    from pmcopilot.api.main import ollama_reachable
    from pmcopilot.observability.alerts import evaluate_rules
    from pmcopilot.store import get_store

    settings = get_settings()
    for r in evaluate_rules(settings, get_store(settings.app_db_path), ollama_reachable(settings)):
        print(f"  [{'ALERT' if r['firing'] else 'ok':5s}] {r['rule']}: {r['message']}")


def cmd_backup(_) -> None:
    from pmcopilot.ops.backup import create_backup

    print(f"Backup: {create_backup(get_settings())}")


def cmd_langfuse_init(_) -> None:
    from pmcopilot.observability.langfuse_setup import init_local_langfuse

    info = init_local_langfuse()
    print("Langfuse configurato. Il .env del progetto ora contiene le chiavi.")
    print(f"  Interfaccia: {info['url']}  (pronta 2-3 minuti dopo l'avvio)")
    print(f"  Email:       {info['email']}")
    print(f"  Password:    {info['password']}")
    print(f"  Segreti in:  {info['infra_env']}")


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pmc", description="Predictive Maintenance Copilot")
    parser.add_argument("--plain-logs", action="store_true", help="log leggibili invece che JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    def train_args(p):
        p.add_argument("--quick", action="store_true", help="meno modelli, più veloce")
        p.add_argument("--split", choices=["temporal", "random"], default="temporal")
        p.add_argument("--promote", choices=["if_better", "always", "never"], default="if_better")

    p = sub.add_parser("setup", help="pipeline completa")
    p.add_argument("--force", action="store_true")
    train_args(p)
    p.set_defaults(func=cmd_setup)
    p = sub.add_parser("data")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_data)
    sub.add_parser("eda").set_defaults(func=cmd_eda)
    p = sub.add_parser("train")
    train_args(p)
    p.set_defaults(func=cmd_train)
    sub.add_parser("score").set_defaults(func=cmd_score)
    p = sub.add_parser("ingest")
    p.add_argument("--full", action="store_true", help="ricostruisce l'indice da zero")
    p.set_defaults(func=cmd_ingest)
    sub.add_parser("chat").set_defaults(func=cmd_chat)

    p = sub.add_parser("eval", help="valutazione e storico")
    p.add_argument("target", choices=["guardrails", "retrieval", "agent", "all", "list", "compare", "gate"])
    p.add_argument("runs", nargs="*", help="per compare: RUN_A RUN_B")
    p.add_argument("--mode", choices=["dense", "bm25", "hybrid", "all"])
    p.add_argument("--with-llm", action="store_true", help="guardrails: valuta anche il classificatore LLM")
    p.add_argument("--limit", type=int)
    p.add_argument("--label")
    p.add_argument("--kind", choices=["agent", "retrieval", "guardrails", "model"])
    p.add_argument("--run", help="gate: run da verificare (default: l'ultimo)")
    p.add_argument("--baseline", help="gate: run di riferimento per i cali massimi")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("simulate", help="aggiunge letture simulate (anche con drift)")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--scenario", default="normal",
                   choices=["normal", "seasonal", "sensor_bias", "late_replacement", "concept"])
    p.add_argument("--intensity", type=float, default=1.0)
    p.add_argument("--ramp", action="store_true", help="drift graduale invece che improvviso")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_simulate)
    p = sub.add_parser("drift")
    p.add_argument("action", choices=["check"])
    p.set_defaults(func=cmd_drift)
    p = sub.add_parser("model")
    p.add_argument("action", choices=["list", "promote"])
    p.add_argument("version", nargs="?")
    p.set_defaults(func=cmd_model)
    p = sub.add_parser("alerts")
    p.add_argument("action", choices=["check"])
    p.set_defaults(func=cmd_alerts)
    sub.add_parser("backup").set_defaults(func=cmd_backup)
    sub.add_parser("langfuse-init").set_defaults(func=cmd_langfuse_init)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    # In chat i log riempirebbero il terminale: solo avvisi ed errori (le trace restano su file).
    level = "WARNING" if args.command == "chat" else settings.log_level
    setup_logging_from_settings(settings, json_format=not args.plain_logs, level=level)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nErrore: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        from pmcopilot.observability.tracing import flush_langfuse

        flush_langfuse(settings)


if __name__ == "__main__":
    main()
