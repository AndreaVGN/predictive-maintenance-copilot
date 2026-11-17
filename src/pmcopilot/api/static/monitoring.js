/* =============================================================================
   monitoring.js — pagina "Monitoraggio" (solo amministratore).

   Risponde alla domanda "posso dormire tranquillo?":
     * Alert        regole valutate ogni minuto dallo scheduler (tasso di errore,
                    latenza p95, picchi di guardrail, feedback negativi, drift,
                    Ollama irraggiungibile). Restano nello storico anche risolti.
     * Drift        il modello ML è stato addestrato su certi dati: se i dati di
                    oggi sono diversi (data drift), se le previsioni si spostano
                    (prediction drift) o se il modello sbaglia di più (concept
                    drift), lo vediamo qui. Ogni controllo è salvato.
     * Simulazione  per provarlo senza aspettare mesi: genera letture finte con
                    uno scenario di deriva e guarda come reagiscono i controlli.
     * Lavori       scoring, drift, alert, reindicizzazione, backup.
     * Modelli      il registry: quale versione è in produzione, promozione e
                    rollback (promuovere una versione vecchia È il rollback).
     * Utilizzo     richieste, errori, latenza, token e costo per giorno.
     * Audit        chi ha fatto cosa: login, approvazioni, promozioni, simulazioni.
   ============================================================================= */

import { css, el, fmt, getJSON, lineChart, postJSON, runTask, show, table } from "./util.js";

const shortDate = (iso) => (iso ? new Date(iso).toLocaleString("it-IT", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) : "–");
const STATUS_TEXT = { ok: "nella norma", warning: "attenzione", alert: "allarme" };
// Per i lavori periodici "allarme" vuol dire "ultima esecuzione fallita": il dettaglio è nel tooltip.
const statusChip = (s) => el("span", { class: `chip status-${s}` }, STATUS_TEXT[s] || s);

const DIAGNOSIS = {
  data_drift: ["Data drift", "La distribuzione di uno o più sensori è cambiata rispetto ai dati di training (PSI alto). Il modello vede input che conosce poco."],
  prediction_drift: ["Prediction drift", "Le probabilità che il modello produce si sono spostate: per esempio molte più macchine sopra soglia."],
  performance_drop: ["Concept drift (calo di prestazioni)", "Con i guasti reali della finestra, il modello trova meno guasti di quanto faceva in test: la relazione tra sensori e guasto è cambiata. Serve riaddestrare."],
};

const SCENARIOS = {
  normal: "Normale: stessi dati del training. Atteso: nessun drift (controllo dei falsi allarmi).",
  seasonal: "Estate: aria e processo più caldi di 6 K. Atteso: data drift sulle temperature.",
  sensor_bias: "Sensore starato: la temperatura dell'aria misurata è +4 K. Atteso: data drift, previsioni spostate e calo di prestazioni.",
  late_replacement: "Utensili sostituiti in ritardo: l'usura sale oltre il solito. Atteso: data drift sull'usura e più previsioni di guasto.",
  concept: "Concept drift: la macchina si guasta con un delta termico più alto di prima. I sensori sembrano normali, ma il modello sbaglia. Atteso: solo calo di prestazioni.",
};

const FEATURE_LABELS = {
  air_temp_k: "Temperatura aria [K]", process_temp_k: "Temperatura processo [K]", rpm: "Giri [rpm]",
  torque_nm: "Coppia [Nm]", wear_x_torque: "Usura × coppia", tool_wear_min: "Usura utensile [min]", temp_diff_k: "Delta termico [K]", power_w: "Potenza [W]",
};

// ============================================================== alert
function alertsPanel(alerts) {
  const open = alerts.filter((a) => !a.resolved_at);
  return el("div", { class: "panel" },
    el("div", { class: "panel-head" }, el("h2", {}, "Alert"),
      el("span", { class: `chip ${open.length ? "status-alert" : "status-ok"}` }, open.length ? `${open.length} ${open.length === 1 ? "aperto" : "aperti"}` : "nessuno aperto")),
    alerts.length
      ? el("div", { class: "table-scroll short" }, table(["Aperto", "Regola", "Gravità", "Messaggio", "Risolto"], alerts.slice(0, 30).map((a) => el("tr", { class: a.resolved_at ? "muted" : "" },
        el("td", {}, shortDate(a.ts)), el("td", {}, a.rule), el("td", {}, statusChip(a.severity === "critical" ? "alert" : "warning")),
        el("td", {}, a.message), el("td", {}, a.resolved_at ? shortDate(a.resolved_at) : "aperto")))))
      : el("p", { class: "note" }, "Nessun alert nello storico. Il webhook (PMC_ALERT_WEBHOOK_URL) riceve gli alert nuovi."));
}

// ============================================================== drift
function driftPanel(reports) {
  if (!reports.length) {
    return el("div", { class: "panel" }, el("h2", {}, "Drift"),
      el("p", { class: "note" }, "Nessun controllo ancora. Lancia una simulazione qui sotto o esegui il lavoro \"drift\"."));
  }
  const last = reports[0];
  const diagnosis = last.prediction?.diagnosis || [];
  const perf = last.performance || {};
  const pred = last.prediction || {};
  const features = Object.entries(last.features || {}).sort((a, b) => b[1].psi - a[1].psi);

  // Grafici: dal più vecchio al più recente.
  const history = [...reports].reverse();
  const labels = history.map((r) => shortDate(r.ts));
  const psiBox = el("div");
  const perfBox = el("div");
  requestAnimationFrame(() => {
    const warn = 0.1, alert = 0.25;
    lineChart(psiBox, {
      labels,
      series: [
        { name: "PSI massimo tra i sensori", color: css("--blue"), values: history.map((r) => Math.max(...Object.values(r.features || {}).map((f) => f.psi))) },
        { name: "PSI delle previsioni", color: css("--lamp-warn"), values: history.map((r) => r.prediction?.psi ?? null) },
      ],
      yMin: 0,
      bands: [
        { from: warn, to: alert, color: "rgba(224,138,0,.10)" },
        { from: alert, to: 99, color: "rgba(208,52,44,.10)" },
      ],
    });
    lineChart(perfBox, {
      labels,
      series: [
        { name: "PR-AUC sulla finestra", color: css("--blue"), values: history.map((r) => r.performance?.pr_auc ?? null) },
        { name: "Recall sulla finestra", color: css("--lamp-ok"), values: history.map((r) => r.performance?.recall ?? null) },
      ],
      yMin: 0, yMax: 1,
    });
  });

  return el("div", { class: "panel drift" },
    el("div", { class: "panel-head" },
      el("h2", {}, "Drift"),
      el("div", {}, statusChip(last.status), el("span", { class: "note" }, ` controllo del ${shortDate(last.ts)}, ${fmt.format(last.n_rows)} letture, modello ${last.model_version}`))),
    diagnosis.length
      ? el("div", { class: "diagnosis" }, diagnosis.map((d) => el("div", {}, el("strong", {}, DIAGNOSIS[d]?.[0] || d), el("p", {}, DIAGNOSIS[d]?.[1] || ""))))
      : el("p", { class: "note" }, "Nessuna deriva: i dati recenti somigliano a quelli di training e il modello rende come in test."),
    el("div", { class: "drift-grid" },
      el("div", {},
        el("h3", {}, "Sensori"),
        table(["Grandezza", "PSI", "Stato", "KS", "p-value", "Media training", "Media ora"], features.map(([k, f]) => el("tr", {},
          el("td", {}, FEATURE_LABELS[k] || k), el("td", { class: "num" }, show(f.psi)), el("td", {}, statusChip(f.status)),
          el("td", { class: "num" }, show(f.ks_stat)), el("td", { class: "num" }, f.ks_pvalue < 0.001 ? "< 0,001" : show(f.ks_pvalue)),
          el("td", { class: "num" }, show(f.mean_ref)), el("td", { class: "num" }, show(f.mean_now)))))),
      el("div", {},
        el("h3", {}, "Previsioni e prestazioni"),
        table(["", "Valore"], [
          ["PSI delle probabilità", show(pred.psi)],
          ["Probabilità media (training → ora)", `${show(pred.mean_ref)} → ${show(pred.mean_now)}`],
          ["Letture sopra soglia ora", show(pred.alarm_rate_now)],
          ["Tasso di guasto (training → ora)", `${show(perf.positive_rate_ref)} → ${show(perf.positive_rate_now)}`],
          ["Guasti nella finestra", show(perf.n_failures)],
          ["PR-AUC sulla finestra", show(perf.pr_auc)],
          ["Recall sulla finestra", show(perf.recall)],
          ["Calo massimo rispetto al test", show(perf.max_drop)],
        ].map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", { class: "num" }, v)))))),
    el("p", { class: "note" }, "PSI sotto 0,1: stabile. Tra 0,1 e 0,25: attenzione. Sopra 0,25: la distribuzione è cambiata. KS è un secondo test sulla stessa domanda: un p-value piccolo dice che le due distribuzioni sono diverse."),
    el("div", { class: "drift-charts" },
      el("div", {}, el("h3", {}, "PSI nel tempo"), psiBox),
      el("div", {}, el("h3", {}, "Prestazioni nel tempo"), perfBox)));
}

function simulatePanel(onDone) {
  const scenario = el("select", {}, Object.keys(SCENARIOS).map((k) => el("option", { value: k }, k)));
  const help = el("p", { class: "note" }, SCENARIOS.normal);
  scenario.addEventListener("change", () => { help.textContent = SCENARIOS[scenario.value]; });
  const hours = el("input", { type: "number", min: 1, max: 720, value: 24 });
  const intensity = el("input", { type: "range", min: 0, max: 3, step: 0.1, value: 1 });
  const intensityOut = el("output", {}, "1");
  intensity.addEventListener("input", () => { intensityOut.textContent = intensity.value; });
  const ramp = el("input", { type: "checkbox" });
  const status = el("p", { class: "note", "aria-live": "polite" });
  const button = el("button", { class: "btn primary", type: "submit" }, "Simula e controlla");
  const form = el("form", { class: "panel" },
    el("h2", {}, "Simula il flusso dei sensori"),
    el("p", { class: "note" }, "Aggiunge letture finte al database (come se fossero passate delle ore), aggiorna lo scoring e lancia il controllo del drift."),
    el("label", { class: "field" }, el("span", {}, "Scenario"), scenario), help,
    el("label", { class: "field" }, el("span", {}, "Ore da simulare"), hours),
    el("label", { class: "field" }, el("span", {}, "Intensità della deriva"), el("div", { class: "inline" }, intensity, intensityOut)),
    el("label", { class: "check" }, ramp, "Graduale: la deriva cresce da 0 al valore scelto"),
    el("div", { class: "actions" }, button), status);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    button.disabled = true;
    status.textContent = "Simulazione in corso…";
    try {
      const r = await postJSON("/ops/simulate", {
        scenario: scenario.value, hours: Number(hours.value), intensity: Number(intensity.value), ramp: ramp.checked,
      });
      const diag = r.drift.diagnosis.length ? r.drift.diagnosis.map((d) => DIAGNOSIS[d]?.[0] || d).join(", ") : "nessuna deriva";
      status.textContent = `Aggiunte ${fmt.format(r.simulation.rows)} letture (${r.simulation.failures} guasti). Esito: ${STATUS_TEXT[r.drift.status] || r.drift.status} (${diag}).`;
      onDone();
    } catch (err) {
      status.textContent = `Errore: ${err.message}`;
    } finally {
      button.disabled = false;
    }
  });
  return form;
}

// ============================================================== lavori
function jobsPanel(jobs, onDone) {
  const every = (s) => (s >= 3600 ? `${fmt.format(s / 3600)} h` : s >= 60 ? `${fmt.format(s / 60)} min` : `${s} s`);
  return el("div", { class: "panel" },
    el("h2", {}, "Lavori periodici"),
    jobs.length
      ? el("div", { class: "table-scroll" }, table(["Lavoro", "Ogni", "Ultima esecuzione", "Esito", "Durata", "Prossima tra", ""], jobs.map((j) => {
        const btn = el("button", { class: "btn ghost small", type: "button" }, "Esegui ora");
        btn.addEventListener("click", async () => {
          btn.disabled = true; btn.textContent = "In corso…";
          try { await postJSON(`/ops/jobs/${j.name}/run`); onDone(); } catch (err) { btn.textContent = err.message; }
        });
        return el("tr", {}, el("td", {}, j.name), el("td", {}, every(j.interval_s)), el("td", {}, shortDate(j.last_run)),
          el("td", { title: j.last_status }, j.last_run ? statusChip(j.last_status === "ok" ? "ok" : "alert") : "mai eseguito"),
          el("td", { class: "num" }, j.last_duration_ms != null ? `${fmt.format(j.last_duration_ms)} ms` : "–"),
          el("td", {}, every(j.next_in_s)), el("td", {}, btn));
      })))
      : el("p", { class: "note" }, "Scheduler disattivato (PMC_SCHEDULER_ENABLED=false)."));
}

// ============================================================== modelli
function modelsPanel(models, onDone) {
  const status = el("p", { class: "note", "aria-live": "polite" });
  const trainBtn = el("button", { class: "btn ghost", type: "button" }, "Riaddestra (rapido)");
  trainBtn.addEventListener("click", async () => {
    trainBtn.disabled = true;
    status.textContent = "Addestramento in corso…";
    try {
      const r = await runTask("/ops/train", { quick: true, split: "temporal", promote_policy: "if_better" },
        (t) => { status.textContent = `Addestramento: ${t.status}…`; });
      status.textContent = `Nuova versione ${r.version}: PR-AUC ${show(r.pr_auc)} contro ${show(r.champion_pr_auc)} del campione, ${r.promoted ? "promossa in produzione" : "resta candidata"}.`;
      onDone();
    } catch (err) {
      status.textContent = `Errore: ${err.message}`;
    } finally {
      trainBtn.disabled = false;
    }
  });
  return el("div", { class: "panel" },
    el("div", { class: "panel-head" }, el("h2", {}, "Registry dei modelli"), trainBtn),
    el("p", { class: "note" }, "Ogni addestramento crea una versione. Promuovere copia quella versione in produzione e la ricarica nell'API senza riavvio. Promuovere una versione precedente è il rollback."),
    status,
    el("div", { class: "table-scroll short" }, table(["Versione", "Data", "Algoritmo", "Split", "PR-AUC", "Recall", "Stato", ""], models.map((m) => {
      const action = m.status === "production" ? el("span", { class: "chip status-ok" }, "in produzione")
        : el("button", {
          class: "btn ghost small", type: "button", onclick: async (e) => {
            e.target.disabled = true;
            try { await postJSON(`/ops/models/${encodeURIComponent(m.version)}/promote`); onDone(); } catch (err) { status.textContent = err.message; }
          },
        }, "Promuovi");
      return el("tr", { class: m.status === "production" ? "selected" : "" },
        el("td", {}, m.version), el("td", {}, shortDate(m.ts)), el("td", {}, m.algorithm), el("td", {}, m.split || "–"),
        el("td", { class: "num" }, show(m.metrics?.pr_auc)), el("td", { class: "num" }, show(m.metrics?.recall)),
        el("td", {}, m.status === "production" ? `produzione${m.promoted_by ? ` (${m.promoted_by})` : ""}` : ({ candidate: "candidato", archived: "archiviato" }[m.status] || m.status)), el("td", {}, action));
    }))));
}

// ============================================================== utilizzo
function usagePanel(days) {
  const chart = el("div");
  requestAnimationFrame(() => lineChart(chart, {
    labels: days.map((d) => d.day.slice(5)),
    series: [
      { name: "Richieste", color: css("--blue"), values: days.map((d) => d.requests) },
      { name: "Errori", color: css("--lamp-alarm"), values: days.map((d) => d.errors) },
    ],
    yMin: 0, height: 180,
  }));
  return el("div", { class: "panel" },
    el("h2", {}, "Utilizzo per giorno"),
    chart,
    days.length ? table(["Giorno", "Richieste", "Errori", "Latenza p95", "Token", "Costo"], [...days].reverse().map((d) => el("tr", {},
      el("td", {}, d.day), el("td", { class: "num" }, show(d.requests)), el("td", { class: "num" }, show(d.errors)),
      el("td", { class: "num" }, `${fmt.format(d.p95_ms / 1000)} s`), el("td", { class: "num" }, show(d.tokens)),
      el("td", { class: "num" }, `${d.cost_usd.toFixed(4)} $`)))) : null);
}

function auditPanel(rows) {
  return el("div", { class: "panel" },
    el("h2", {}, "Registro delle azioni (audit)"),
    rows.length
      ? el("div", { class: "table-scroll short" }, table(["Quando", "Utente", "Azione", "Oggetto", "Dettagli"], rows.map((r) => el("tr", {},
        el("td", {}, shortDate(r.ts)), el("td", {}, r.user), el("td", {}, r.action), el("td", {}, r.target || ""),
        el("td", { class: "mono" }, r.details ? JSON.stringify(r.details) : "")))))
      : el("p", { class: "note" }, "Nessuna azione registrata."));
}

// ============================================================== pagina
export async function loadMonitoring(box) {
  if (!box.children.length) box.replaceChildren(el("p", { class: "note" }, "Carico…"));
  const reload = () => loadMonitoring(box);
  try {
    const [alerts, drift, jobs, models, days, audit] = await Promise.all([
      getJSON("/ops/alerts"), getJSON("/ops/drift?limit=60"), getJSON("/ops/jobs"),
      getJSON("/ops/models"), getJSON("/metrics/daily?days=14"), getJSON("/ops/audit?limit=60")]);
    box.replaceChildren(
      alertsPanel(alerts),
      driftPanel(drift),
      el("div", { class: "mon-grid" }, simulatePanel(reload), jobsPanel(jobs, reload)),
      modelsPanel(models, reload),
      el("div", { class: "mon-grid" }, usagePanel(days), auditPanel(audit)),
    );
  } catch (err) {
    box.replaceChildren(el("div", { class: "error-box" }, `Impossibile leggere il monitoraggio: ${err.message}`));
  }
}
