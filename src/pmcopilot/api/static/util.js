/* =============================================================================
   util.js — funzioni condivise dalla console: DOM, chiamate all'API, grafici.

   SICUREZZA
     Il testo che arriva dall'LLM o dal database non viene MAI inserito come
     HTML grezzo: passa da escapeHtml() o da textContent. Altrimenti una
     risposta contenente "<script>" verrebbe eseguita (attacco XSS).
   ============================================================================= */

export const $ = (sel, root = document) => root.querySelector(sel);
export const fmt = new Intl.NumberFormat("it-IT", { maximumFractionDigits: 1 });
export const fmt3 = new Intl.NumberFormat("it-IT", { maximumFractionDigits: 3 });
export const pct = (p) => (p == null ? "–" : `${fmt.format(p * 100)}%`);
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

/* Crea un elemento: el("button", {class: "btn", onclick: f}, "Testo", altroNodo) */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;            // solo con contenuto già "escapato"
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// ------------------------------------------------------------------ API
/* Se la sessione scade, il server risponde 401: chi ha registrato onUnauthorized
   (app.js) mostra di nuovo la schermata di login. */
export const api = { onUnauthorized: () => {} };

async function check(res) {
  if (res.status === 401) {
    api.onUnauthorized();
    throw new Error("Sessione scaduta: accedi di nuovo.");
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(typeof detail.detail === "string" ? detail.detail : `Errore ${res.status}`);
  }
  return res;
}

export async function getJSON(url) {
  return (await check(await fetch(url))).json();
}

export async function postJSON(url, body = {}) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return (await check(res)).json();
}

/* Legge uno stream Server-Sent Events da una POST.
   EventSource supporta solo GET, quindi usiamo fetch() e leggiamo il corpo a
   pezzi: ogni evento è una riga "data: {...}" seguita da una riga vuota. */
export async function streamSSE(url, body, onEvent) {
  const res = await check(await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }));
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

/* Avvia un lavoro lungo (valutazione, training) e aspetta che finisca. */
export async function runTask(url, body, onProgress) {
  const task = await postJSON(url, body);
  for (;;) {
    await new Promise((r) => setTimeout(r, 1500));
    const state = await getJSON(`/ops/tasks/${task.id}`);
    onProgress?.(state);
    if (state.status === "completato") return state.result;
    if (state.status === "errore") throw new Error(state.error);
  }
}

// ------------------------------------------------------------- Markdown
/* Markdown minimale per le risposte dell'LLM: titoli, elenchi, grassetto,
   corsivo e codice inline. Prima si fa l'escape, poi si aggiungono i tag. */
export function renderMarkdown(text) {
  const inline = (s) => escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of String(text || "").split("\n")) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) { closeList(); continue; }
    if ((m = line.match(/^#{1,6}\s+(.*)/))) { closeList(); out.push(`<h4>${inline(m[1])}</h4>`); continue; }
    if ((m = line.match(/^\s*[-*•]\s+(.*)/))) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(m[1])}</li>`); continue;
    }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(m[1])}</li>`); continue;
    }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return out.join("");
}

// -------------------------------------------------------------- tabelle
export const LABELS = {
  n_questions: "Domande", n_cases: "Casi", input_cases: "Casi di ingresso", sql_cases: "Query SQL",
  input_detection_rate: "Attacchi bloccati (ingresso)", input_false_positive_rate: "Domande legittime bloccate",
  sql_detection_rate: "Query pericolose bloccate", sql_false_positive_rate: "Query legittime bloccate",
  llm_classifier: "Classificatore LLM", classifier_latency_ms_avg: "Latenza classificatore (ms)",
  "hit@1": "Fonte giusta al 1° posto (hit@1)", "hit@4": "Fonte giusta nei primi 4 (hit@4)", mrr: "MRR",
  mrr_semantica: "MRR domande semantiche", mrr_numerica: "MRR domande con numeri", mrr_codice: "MRR codici di allarme",
  pass_rate: "Casi superati", route_accuracy: "Routing corretto", tool_selection_accuracy: "Tool scelti correttamente",
  keyword_hit_rate: "Parole chiave presenti", faithfulness_avg_1to5: "Fedeltà media (1-5)",
  approval_gate_ok: "Approvazione richiesta quando serve", errors: "Errori",
  latency_p50_ms: "Latenza p50 (ms)", latency_p95_ms: "Latenza p95 (ms)", avg_llm_calls: "Chiamate LLM medie",
  avg_tokens: "Token medi per domanda", cost_usd_total: "Costo totale (USD)",
  pr_auc: "PR-AUC", roc_auc: "ROC-AUC", precision: "Precision", recall: "Recall", f1: "F1",
  threshold: "Soglia", promoted: "Promosso", champion_pr_auc: "PR-AUC del campione",
};
export const label = (k) => LABELS[k] || (k.startsWith("pass_") ? `Superati: ${k.slice(5)}` : k);

export function show(v) {
  if (typeof v === "boolean") return v ? "sì" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? fmt.format(v) : fmt3.format(v);
  return String(v ?? "–");
}

export function kvTable(obj) {
  return el("table", {}, el("tbody", {}, Object.entries(obj).map(([k, v]) =>
    el("tr", {}, el("th", {}, label(k)), el("td", { class: "num" }, show(v))))));
}

export function table(headers, rows) {
  return el("table", {}, el("thead", {}, el("tr", {}, headers.map((h) => el("th", {}, h)))),
    el("tbody", {}, rows));
}

// -------------------------------------------------------------- grafici
/* Serie singola compatta (mini-grafici dell'impianto). */
export function drawSeries(canvas, values, { color, threshold = null, marks = [] } = {}) {
  // <canvas> disegna pixel: per uno schermo Retina moltiplichiamo per devicePixelRatio.
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const nums = values.filter((v) => v != null);
  if (!nums.length) {
    ctx.fillStyle = css("--steel"); ctx.font = "13px Barlow, sans-serif";
    ctx.fillText("Nessun dato nel periodo", 10, h / 2);
    return;
  }
  let lo = Math.min(...nums, threshold ?? Infinity), hi = Math.max(...nums, threshold ?? -Infinity);
  if (hi === lo) { hi += 1; lo -= 1; }
  const pad = 8;
  const x = (i) => pad + (i / Math.max(values.length - 1, 1)) * (w - 2 * pad);
  const y = (v) => h - pad - ((v - lo) / (hi - lo)) * (h - 2 * pad);
  if (threshold != null) {
    ctx.strokeStyle = css("--steel"); ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, y(threshold)); ctx.lineTo(w - pad, y(threshold)); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round";
  ctx.beginPath();
  let pen = false;
  values.forEach((v, i) => {
    if (v == null) { pen = false; return; }
    if (pen) ctx.lineTo(x(i), y(v)); else { ctx.moveTo(x(i), y(v)); pen = true; }
  });
  ctx.stroke();
  ctx.fillStyle = css("--lamp-alarm");
  for (const i of marks) {
    if (values[i] == null) continue;
    ctx.beginPath(); ctx.arc(x(i), y(values[i]), 4, 0, 2 * Math.PI); ctx.fill();
  }
}

/* Grafico a linee con assi, più serie, legenda e tooltip al passaggio del mouse.
   series = [{name, values, color}], labels = etichette sull'asse x (una per punto).
   bands = [{from, to, color}] fasce orizzontali (es. zone PSI ok/attenzione/allarme). */
export function lineChart(container, { labels, series, yMin = null, yMax = null, format = show, bands = [], height = 220 }) {
  const wrap = el("div", { class: "chart" });
  const canvas = el("canvas", { role: "img", "aria-label": series.map((s) => s.name).join(", "), style: `height:${height}px` });
  const tip = el("div", { class: "chart-tip", hidden: true });
  const legend = el("div", { class: "chart-legend" }, series.map((s) =>
    el("span", {}, el("i", { style: `background:${s.color}` }), s.name)));
  wrap.append(legend, canvas, tip);
  container.replaceChildren(wrap);

  const draw = (hover = null) => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const all = series.flatMap((s) => s.values).filter((v) => v != null);
    if (!all.length || !labels.length) {
      ctx.fillStyle = css("--steel"); ctx.font = "13px Barlow, sans-serif";
      ctx.fillText("Nessun dato: esegui almeno una valutazione.", 12, h / 2);
      return null;
    }
    let lo = yMin ?? Math.min(...all), hi = yMax ?? Math.max(...all);
    if (hi === lo) { hi += 1; lo -= 1; }
    const left = 48, right = 12, top = 10, bottom = 24;
    const x = (i) => left + (labels.length === 1 ? (w - left - right) / 2 : (i / (labels.length - 1)) * (w - left - right));
    const y = (v) => top + (1 - (v - lo) / (hi - lo)) * (h - top - bottom);
    for (const b of bands) {
      ctx.fillStyle = b.color;
      const y1 = y(Math.min(b.to, hi)), y2 = y(Math.max(b.from, lo));
      if (y2 > y1) ctx.fillRect(left, y1, w - left - right, y2 - y1);
    }
    ctx.strokeStyle = css("--rule"); ctx.fillStyle = css("--steel"); ctx.font = "11px Barlow, sans-serif";
    ctx.lineWidth = 1;
    for (let k = 0; k <= 4; k++) {
      const v = lo + (k / 4) * (hi - lo);
      ctx.beginPath(); ctx.moveTo(left, y(v)); ctx.lineTo(w - right, y(v)); ctx.stroke();
      ctx.fillText(format(v), 4, y(v) + 4);
    }
    const step = Math.max(1, Math.ceil(labels.length / 6));
    labels.forEach((lab, i) => { if (i % step === 0) ctx.fillText(lab, x(i) - 14, h - 6); });
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
      let pen = false;
      s.values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        if (pen) ctx.lineTo(x(i), y(v)); else { ctx.moveTo(x(i), y(v)); pen = true; }
      });
      ctx.stroke();
      ctx.fillStyle = s.color;
      s.values.forEach((v, i) => { if (v != null) { ctx.beginPath(); ctx.arc(x(i), y(v), i === hover ? 5 : 3, 0, 2 * Math.PI); ctx.fill(); } });
    }
    if (hover != null) {
      ctx.strokeStyle = css("--steel"); ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x(hover), top); ctx.lineTo(x(hover), h - bottom); ctx.stroke(); ctx.setLineDash([]);
    }
    return x;
  };
  let xOf = null;
  requestAnimationFrame(() => { xOf = draw(); });
  canvas.addEventListener("mousemove", (e) => {
    if (!xOf || !labels.length) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    let best = 0;
    labels.forEach((_, i) => { if (Math.abs(xOf(i) - mx) < Math.abs(xOf(best) - mx)) best = i; });
    xOf = draw(best);
    tip.hidden = false;
    tip.replaceChildren(el("strong", {}, labels[best]), ...series.map((s) => el("div", {}, `${s.name}: ${format(s.values[best])}`)));
    tip.style.left = `${Math.min(xOf(best) + 12, rect.width - 180)}px`;
  });
  canvas.addEventListener("mouseleave", () => { tip.hidden = true; xOf = draw(); });
}

export const SERIES_COLORS = ["--blue", "--lamp-warn", "--lamp-ok", "--violet"];
