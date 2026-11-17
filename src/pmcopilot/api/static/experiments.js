/* =============================================================================
   experiments.js — pagina "Esperimenti" (solo amministratore).

   A COSA SERVE
     Cambi il prompt, il modello LLM, il retriever o riaddestri il modello ML:
     è andata meglio o peggio? Qui ogni valutazione è un "run" salvato nel
     database (tabella eval_runs) con:
       * il riassunto delle metriche (pass rate, MRR, PR-AUC, latenza, costo…)
       * la configurazione al momento del run (LLM, versione del prompt, …)
       * i risultati caso per caso (tabella eval_cases)
     Lo storico non si cancella: puoi confrontare un run di oggi con uno di
     tre mesi fa.

   COSA SI FA IN QUESTA PAGINA
     1. Lanciare una nuova valutazione (gira in background sul server).
     2. Vedere l'andamento di una metrica nel tempo.
     3. Scegliere due run (A = prima, B = dopo) e confrontarli:
        differenze delle metriche, parametri cambiati, regressioni caso per
        caso, diff del prompt.
     4. Applicare il quality gate: le soglie di eval/datasets/quality_gates.json
        che in CI decidono se un cambiamento può essere rilasciato.
   ============================================================================= */

import {
  SERIES_COLORS, css, el, escapeHtml, fmt, getJSON, kvTable, label, lineChart, runTask, show, table,
} from "./util.js";

const state = { kind: "agent", runs: [], a: null, b: null, metric: null };

// Le metriche mostrate in tabella per ogni tipo di run (le altre sono nel confronto).
const KEY_METRICS = {
  agent: ["pass_rate", "faithfulness_avg_1to5", "latency_p95_ms", "cost_usd_total"],
  retrieval: ["hit@1", "hit@4", "mrr"],
  guardrails: ["input_detection_rate", "input_false_positive_rate", "sql_detection_rate"],
  model: ["pr_auc", "recall", "precision", "promoted"],
};
const DEFAULT_METRIC = { agent: "pass_rate", retrieval: "mrr", guardrails: "input_detection_rate", model: "pr_auc" };
const AGENT_CATEGORIES = ["sql", "rag", "ml", "what_if", "multi_tool", "chart", "approval", "security", "robustness", "out_of_scope"];

const shortDate = (iso) => new Date(iso).toLocaleString("it-IT", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });

/* Cosa distingue un run dagli altri: per l'agente il modello e il prompt,
   per il retrieval la modalità di ricerca, per il modello ML l'algoritmo. */
function configSummary(run) {
  const c = run.config || {};
  if (run.kind === "agent") return `${c.llm || "?"} · prompt ${run.prompt_version || c.prompt_version || "?"}`;
  if (run.kind === "retrieval") return `${c.retriever_mode || "?"} · ${c.embedding || ""}`;
  if (run.kind === "guardrails") return run.summary?.llm_classifier ? "regole + classificatore LLM" : "solo regole";
  return `${c.algorithm || "?"} · split ${c.split || "?"}`;
}

// ======================================================== 1. nuovo run
function runForm(kind, onDone) {
  const status = el("p", { class: "note", "aria-live": "polite" });
  const labelInput = el("input", { placeholder: "es. prompt con regola 8, qwen3:14b…", maxlength: 120 });
  const fields = [el("label", { class: "field" }, el("span", {}, "Etichetta (facoltativa)"), labelInput)];
  let extra = () => ({});

  if (kind === "agent") {
    const limit = el("input", { type: "number", min: 1, max: 500, placeholder: "tutti (30)" });
    const boxes = AGENT_CATEGORIES.map((c) => el("label", { class: "check" }, el("input", { type: "checkbox", value: c }), c));
    fields.push(el("label", { class: "field" }, el("span", {}, "Numero di casi"), limit),
      el("fieldset", { class: "field" }, el("legend", {}, "Solo queste categorie (nessuna = tutte)"), el("div", { class: "checks" }, boxes)));
    extra = () => {
      const categories = boxes.map((b) => b.firstChild).filter((i) => i.checked).map((i) => i.value);
      return { limit: limit.value ? Number(limit.value) : null, categories: categories.length ? categories : null };
    };
  } else if (kind === "retrieval") {
    const mode = el("select", {}, ["hybrid", "dense", "bm25", "all"].map((m) =>
      el("option", { value: m }, m === "all" ? "tutte e tre (un run per modalità)" : m)));
    fields.push(el("label", { class: "field" }, el("span", {}, "Modalità di ricerca"), mode));
    extra = () => ({ mode: mode.value });
  } else if (kind === "guardrails") {
    const withLlm = el("input", { type: "checkbox" });
    fields.push(el("label", { class: "check" }, withLlm, "Valuta anche il classificatore LLM (più lento)"));
    extra = () => ({ with_llm: withLlm.checked });
  } else {
    const quick = el("input", { type: "checkbox", checked: true });
    const split = el("select", {}, el("option", { value: "temporal" }, "temporale (realistico)"), el("option", { value: "random" }, "casuale"));
    const policy = el("select", {},
      el("option", { value: "if_better" }, "promuovi solo se batte il campione"),
      el("option", { value: "never" }, "non promuovere (resta candidato)"),
      el("option", { value: "always" }, "promuovi comunque"));
    fields.length = 0;
    fields.push(el("label", { class: "check" }, quick, "Addestramento rapido (meno combinazioni)"),
      el("label", { class: "field" }, el("span", {}, "Divisione train/test"), split),
      el("label", { class: "field" }, el("span", {}, "Promozione"), policy));
    extra = () => ({ quick: quick.checked, split: split.value, promote_policy: policy.value });
  }

  const button = el("button", { class: "btn primary", type: "submit" }, kind === "model" ? "Riaddestra" : "Esegui valutazione");
  const hint = {
    agent: "Fa girare l'agente sui casi del golden set e li fa giudicare da un LLM. Con Ollama locale servono alcuni minuti.",
    retrieval: "Misura se la ricerca trova il manuale giusto (41 domande). Pochi secondi.",
    guardrails: "Prova i filtri su attacchi e domande legittime. Istantaneo con le sole regole.",
    model: "Riaddestra il modello di previsione e lo confronta con quello in produzione sullo stesso test set.",
  }[kind];

  const form = el("form", { class: "panel run-form" },
    el("h2", {}, "Nuovo run"), el("p", { class: "note" }, hint), fields, el("div", { class: "actions" }, button), status);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    button.disabled = true;
    status.textContent = "Avvio…";
    try {
      const url = kind === "model" ? "/ops/train" : "/experiments/run";
      const body = kind === "model" ? extra() : { kind, label: labelInput.value.trim() || null, ...extra() };
      const t0 = Date.now();
      await runTask(url, body, (task) => {
        status.textContent = `${task.status}… ${Math.round((Date.now() - t0) / 1000)} s`;
      });
      status.textContent = "Completato.";
      onDone();
    } catch (err) {
      status.textContent = `Errore: ${err.message}`;
    } finally {
      button.disabled = false;
    }
  });
  return form;
}

// ======================================================== 2. andamento
function trendPanel(trend) {
  const box = el("div", { class: "panel" });
  const numeric = [...new Set(trend.flatMap((r) => Object.entries(r.summary || {})
    .filter(([, v]) => typeof v === "number").map(([k]) => k)))];
  if (!state.metric || !numeric.includes(state.metric)) state.metric = numeric.includes(DEFAULT_METRIC[state.kind]) ? DEFAULT_METRIC[state.kind] : numeric[0];
  const select = el("select", { "aria-label": "Metrica" }, numeric.map((k) =>
    el("option", { value: k, selected: k === state.metric }, label(k))));
  const chartBox = el("div");
  const caption = el("p", { class: "note" });
  const draw = () => {
    state.metric = select.value;
    // Per il retrieval una linea per modalità: confrontare dense con hybrid ha senso, mescolarli no.
    const groupKey = (r) => (state.kind === "retrieval" ? (r.retriever_mode || "run") : "run");
    const groups = [...new Set(trend.map(groupKey))];
    const labels = trend.map((r) => shortDate(r.ts));
    const series = groups.map((g, i) => ({
      name: g === "run" ? label(state.metric) : g,
      color: css(SERIES_COLORS[i % SERIES_COLORS.length]),
      values: trend.map((r) => (groupKey(r) === g ? r.summary?.[state.metric] ?? null : null)),
    }));
    const isRate = /rate|accuracy|hit@|mrr|pr_auc|roc_auc|recall|precision|f1|pass_/.test(state.metric);
    lineChart(chartBox, { labels, series, yMin: isRate ? 0 : null, yMax: isRate ? 1 : null });
    caption.textContent = trend.length
      ? `${trend.length} run, dal più vecchio al più recente. Passa il mouse sui punti per i valori.`
      : "";
  };
  select.addEventListener("change", draw);
  box.append(el("div", { class: "panel-head" }, el("h2", {}, "Andamento nel tempo"), select), chartBox, caption);
  requestAnimationFrame(draw);
  return box;
}

// ============================================================ 3. elenco
function runsPanel(onCompare, onGate) {
  const keys = KEY_METRICS[state.kind];
  const rows = state.runs.map((r) => el("tr", { class: r.id === state.a || r.id === state.b ? "selected" : "" },
    el("td", {}, el("input", { type: "radio", name: "run-a", "aria-label": `A: ${r.id}`, checked: r.id === state.a, onchange: () => { state.a = r.id; } })),
    el("td", {}, el("input", { type: "radio", name: "run-b", "aria-label": `B: ${r.id}`, checked: r.id === state.b, onchange: () => { state.b = r.id; } })),
    el("td", {}, shortDate(r.ts)),
    el("td", {}, el("div", {}, r.label || r.id), el("small", { class: "note" }, configSummary(r))),
    keys.map((k) => el("td", { class: "num" }, show(r.summary?.[k]))),
    el("td", {}, r.user || "–")));
  return el("div", { class: "panel" },
    el("div", { class: "panel-head" },
      el("h2", {}, `Run salvati (${state.runs.length})`),
      el("div", { class: "actions" },
        el("button", { class: "btn ghost", type: "button", onclick: onGate }, "Quality gate su B"),
        el("button", { class: "btn primary", type: "button", onclick: onCompare }, "Confronta A → B"))),
    el("p", { class: "note" }, "A è il riferimento (prima del cambiamento), B il candidato (dopo). Il gate controlla le soglie su B e, se impostato, il calo massimo rispetto ad A."),
    state.runs.length
      ? el("div", { class: "table-scroll" }, table(["A", "B", "Data", "Run", ...keys.map(label), "Utente"], rows))
      : el("p", { class: "note" }, "Nessun run ancora: lanciane uno qui sopra o da terminale con pmc eval."));
}

// =========================================================== 4. confronto
const mark = (passed) => (passed == null ? "–" : passed ? "✓" : "✗");

function caseDetail(kind, a, b) {
  if (kind === "agent") {
    const side = (c, name) => el("div", { class: "case-side" },
      el("h4", {}, `${name}: ${mark(c.passed)}`),
      el("p", { class: "note" }, `route ${c.route ?? "–"} · tool ${(c.tools_used || []).join(", ") || "–"} · fedeltà ${c.faithfulness ?? "–"} · ${fmt.format((c.latency_ms || 0) / 1000)} s`),
      c.error ? el("p", { class: "error-box" }, c.error) : el("div", { class: "answer", html: escapeHtml(c.answer || "").replace(/\n/g, "<br>") }),
      c.judge_reasoning ? el("p", { class: "note" }, `Giudice: ${c.judge_reasoning}`) : null);
    return el("div", { class: "case-compare" }, side(a, "A"), side(b, "B"));
  }
  if (kind === "retrieval") {
    return el("p", { class: "note" }, `Atteso ${b.expected}. A: posizione ${a.rank ?? "non trovato"} (${(a.retrieved || []).join(", ")}). B: posizione ${b.rank ?? "non trovato"} (${(b.retrieved || []).join(", ")}).`);
  }
  if (kind === "guardrails") {
    return el("p", { class: "note" }, `Atteso: ${b.expected === "block" ? "blocco" : "consentito"}. A: ${a.blocked ? "bloccato" : "consentito"} ${(a.flags || []).join(", ")}. B: ${b.blocked ? "bloccato" : "consentito"} ${(b.flags || []).join(", ")}.`);
  }
  return el("pre", {}, JSON.stringify({ A: a, B: b }, null, 2));
}

function comparePanel(cmp) {
  const verdictClass = { meglio: "better", peggio: "worse", uguale: "" };
  const counts = cmp.case_counts;
  // Prima le metriche principali del tipo di run, poi le altre in ordine alfabetico.
  const keys = KEY_METRICS[cmp.kind] || [];
  const rank = (m) => (keys.includes(m.metric) ? keys.indexOf(m.metric) : 100);
  const metrics = [...cmp.metrics].sort((x, y) => rank(x) - rank(y));
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  const metricRows = metrics.map((m) => el("tr", {},
    el("th", {}, label(m.metric)), el("td", { class: "num" }, show(m.a)), el("td", { class: "num" }, show(m.b)),
    el("td", { class: `num ${verdictClass[m.verdict] || ""}` }, m.delta == null ? "" : `${m.delta > 0 ? "+" : ""}${show(m.delta)}`),
    el("td", { class: verdictClass[m.verdict] || "" }, m.verdict || "")));
  const configRows = cmp.config_changes.map((c) => el("tr", {},
    el("th", {}, c.key), el("td", {}, show(c.a)), el("td", {}, show(c.b))));
  const caseRows = cmp.cases.map((c) => {
    const detail = el("tr", { class: "case-detail", hidden: true }, el("td", { colspan: 4 }, caseDetail(cmp.kind, c.a, c.b)));
    const row = el("tr", { class: `case ${c.change}`, tabindex: 0, onclick: () => { detail.hidden = !detail.hidden; } },
      el("td", {}, el("span", { class: `chip ${c.change}` }, c.change)),
      el("td", {}, c.question || c.id),
      el("td", { class: "num" }, mark(c.a.passed)), el("td", { class: "num" }, mark(c.b.passed)));
    row.addEventListener("keydown", (e) => { if (e.key === "Enter") detail.hidden = !detail.hidden; });
    return [row, detail];
  });
  return el("div", { class: "panel compare" },
    el("h2", {}, "Confronto"),
    el("div", { class: "compare-head" },
      el("div", {}, el("small", {}, "A, riferimento"), el("strong", {}, cmp.a.label || cmp.a.id), el("span", { class: "note" }, shortDate(cmp.a.ts))),
      el("span", { class: "arrow", "aria-hidden": "true" }, "→"),
      el("div", {}, el("small", {}, "B, candidato"), el("strong", {}, cmp.b.label || cmp.b.id), el("span", { class: "note" }, shortDate(cmp.b.ts)))),
    el("div", { class: "case-counts" },
      el("span", { class: "chip regressione" }, plural(counts.regressioni, "regressione", "regressioni")),
      el("span", { class: "chip miglioramento" }, plural(counts.miglioramenti, "miglioramento", "miglioramenti")),
      el("span", { class: "chip invariato" }, plural(counts.invariati, "invariato", "invariati")),
      counts.solo_in_a || counts.solo_in_b ? el("span", { class: "note" }, `${counts.solo_in_a} casi solo in A, ${counts.solo_in_b} solo in B`) : null),
    el("div", { class: "compare-grid" },
      el("div", {}, el("h3", {}, "Metriche"), table(["", "A", "B", "Δ", ""], metricRows)),
      el("div", {}, el("h3", {}, "Cosa è cambiato nella configurazione"),
        configRows.length ? table(["Parametro", "A", "B"], configRows) : el("p", { class: "note" }, "Stessa configurazione: le differenze vengono dal non determinismo del modello o dai dati."))),
    cmp.prompt_diff ? [el("h3", {}, "Diff del prompt"), el("pre", { class: "diff" }, cmp.prompt_diff.split("\n").map((line) =>
      el("span", { class: line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "" }, `${line}\n`)))] : null,
    el("h3", {}, "Caso per caso"),
    el("p", { class: "note" }, "Prima le regressioni. Clicca una riga per vedere il dettaglio di A e B."),
    caseRows.length ? el("div", { class: "table-scroll" }, table(["Esito", "Caso", "A", "B"], caseRows)) : el("p", { class: "note" }, "Nessun caso in comune."));
}

function gatePanel(gate) {
  return el("div", { class: `panel gate ${gate.passed ? "pass" : "fail"}` },
    el("h2", {}, gate.passed ? "Quality gate superato" : "Quality gate NON superato"),
    el("p", { class: "note" }, `Run ${gate.run_id}. In CI il comando pmc eval gate ${gate.passed ? "esce con 0 e il rilascio prosegue" : "esce con 1 e blocca il rilascio"}.`),
    gate.checks.length
      ? table(["Regola", "Valore", ""], gate.checks.map((c) => el("tr", {},
        el("td", {}, c.rule), el("td", { class: "num" }, show(c.value)), el("td", { class: c.ok ? "better" : "worse" }, c.ok ? "ok" : "fallita"))))
      : el("p", { class: "note" }, "Nessuna regola definita per questo tipo in quality_gates.json."));
}

// ============================================================== pagina
async function modelExtras() {
  try {
    const model = await getJSON("/model");
    const card = model.card || {};
    return el("div", { class: "panel" },
      el("h2", {}, "Modello in produzione"),
      el("p", { class: "note" }, `${card.algorithm || "–"}, feature ${card.feature_set || "–"}, soglia ${show(card.threshold)}, split ${card.split || "–"}.`),
      kvTable(card.test_metrics || {}),
      model.figures?.length ? el("div", { class: "figures" }, model.figures.map((f) =>
        el("figure", {}, el("img", { src: `/reports/figures/${f}`, alt: f, loading: "lazy" }), el("figcaption", {}, f)))) : null);
  } catch {
    return null;
  }
}

export async function loadExperiments(box) {
  const tabs = document.querySelectorAll("#exp-tabs [role=tab]");
  for (const tab of tabs) {
    tab.onclick = () => {
      state.kind = tab.dataset.kind;
      Object.assign(state, { a: null, b: null, metric: null });
      for (const t of tabs) t.setAttribute("aria-selected", String(t === tab));
      loadExperiments(box);
    };
  }
  box.replaceChildren(el("p", { class: "note" }, "Carico i run…"));
  const kind = state.kind;
  try {
    const [runs, trend] = await Promise.all([
      getJSON(`/experiments/runs?kind=${kind}`), getJSON(`/experiments/trend?kind=${kind}`)]);
    if (kind !== state.kind) return;  // l'utente ha cambiato scheda nel frattempo
    state.runs = runs;
    // Default: B = ultimo run, A = penultimo. Il caso tipico è "com'è andato l'ultimo cambiamento?".
    if (!runs.some((r) => r.id === state.b)) state.b = runs[0]?.id ?? null;
    if (!runs.some((r) => r.id === state.a)) state.a = runs[1]?.id ?? null;

    const result = el("div", { class: "exp-result", "aria-live": "polite" });
    const reload = () => loadExperiments(box);
    const compare = async () => {
      if (!state.a || !state.b || state.a === state.b) {
        result.replaceChildren(el("p", { class: "error-box" }, "Scegli due run diversi: A nella prima colonna, B nella seconda."));
        return;
      }
      result.replaceChildren(el("p", { class: "note" }, "Confronto…"));
      try {
        result.replaceChildren(comparePanel(await getJSON(`/experiments/compare?a=${encodeURIComponent(state.a)}&b=${encodeURIComponent(state.b)}`)));
      } catch (err) {
        result.replaceChildren(el("p", { class: "error-box" }, err.message));
      }
    };
    const gate = async () => {
      if (!state.b) return;
      const qs = new URLSearchParams({ kind, run: state.b });
      if (state.a && state.a !== state.b) qs.set("baseline", state.a);
      try {
        result.replaceChildren(gatePanel(await getJSON(`/experiments/gate?${qs}`)));
      } catch (err) {
        result.replaceChildren(el("p", { class: "error-box" }, err.message));
      }
    };

    box.replaceChildren(
      el("div", { class: "exp-top" }, runForm(kind, reload), trendPanel(trend)),
      runsPanel(compare, gate),
      result,
      ...(kind === "model" ? [await modelExtras()].filter(Boolean) : []),
    );
    if (state.a && state.b) compare();
  } catch (err) {
    box.replaceChildren(el("div", { class: "error-box" }, `Impossibile leggere i run: ${err.message}`));
  }
}
