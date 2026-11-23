"use strict";

const K_VALUES = [1, 3, 5, 10];
const DEFAULT_Q = "Kapan PDI dibubarkan?";
const PAPER = [
  { k: 1, rougeL: 0.0872, bleu: 0.0213, bert: 0.3117, acc: 0.67 },
  { k: 3, rougeL: 0.0867, bleu: 0.019, bert: 0.3105, acc: 0.7 },
  { k: 5, rougeL: 0.0863, bleu: 0.0168, bert: 0.3111, acc: 0.76 },
  { k: 10, rougeL: 0.0739, bleu: 0.0167, bert: 0.2968, acc: 0.36 },
];
const STATUS = {
  s: { cls: "supported", label: "matches the passage", ok: true },
  w: { cls: "weak", label: "loosely matches", ok: null },
  u: { cls: "unsupported", label: "doesn't match", ok: false },
  x: { cls: "invented", label: "points to nothing", ok: false },
};
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = { questions: [], passages: [], qi: 0, k: 3, tab: "explore" };
const $ = (sel) => document.querySelector(sel);

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

const current = () => state.questions[state.qi];
const run = (q = current(), k = state.k) => q.k[String(k)];
const isHallucinatedTrick = (q, r) => !q.ans && !r.idk;

// ── Answer with citation chips ─────────────────────────────────────────
function citeChips(answer, r, q, k) {
  return esc(answer).replace(/\[(\d+)\]/g, (_, n) => {
    let code = r.c[n];
    let cls = code ? STATUS[code].cls : "pending";
    if (Number(n) > k) cls = "invented";
    if (q && isHallucinatedTrick(q, r) && cls === "supported") cls = "pending";
    return `<span class="cite cite-${cls}">${n}</span>`;
  });
}

let typingTimer = null;
function renderAnswer(q, r, k) {
  clearInterval(typingTimer);
  const box = $("#answer-text");
  const full = r.a;
  if (reduceMotion) {
    box.innerHTML = citeChips(full, r, q, k);
    return;
  }
  // Short typewriter reveal so it feels like the model is answering.
  let i = 0;
  const step = Math.max(2, Math.ceil(full.length / 40));
  typingTimer = setInterval(() => {
    i = Math.min(full.length, i + step);
    if (i >= full.length) {
      clearInterval(typingTimer);
      box.innerHTML = citeChips(full, r, q, k);
    } else {
      box.innerHTML = esc(full.slice(0, i)) + '<span class="cursor"></span>';
    }
  }, 25);
}

// ── Checks ────────────────────────────────────────────────────────────
function checkRow(ok, title, detail = "") {
  const icon = ok === true ? "✓" : ok === false ? "✗" : "•";
  const cls = ok === true ? "ok" : ok === false ? "bad" : "info";
  return `<div class="check check-${cls}"><b>${icon}</b><div><strong>${title}</strong>${detail ? `<span>${detail}</span>` : ""}</div></div>`;
}

function renderChecks(q, r, k) {
  const rows = [];
  const cited = Object.keys(r.c);
  if (!q.ans) {
    rows.push(checkRow(null, "Trick question", "The answer isn't in any of the 358 documents."));
    rows.push(r.idk
      ? checkRow(true, "Admitted it doesn't know", "This is the right answer here.")
      : checkRow(false, "Hallucinated an answer", "It answered anyway, so whatever it said is made up."));
  } else {
    rows.push(checkRow(null, "Expected answer", esc(q.gold)));
    if (r.idk) rows.push(checkRow(false, "Said it doesn't know", "The answer was in the documents."));
    if (!r.d.includes(q.doc)) {
      rows.push(checkRow(false, "Source passage not retrieved", "Search didn't find it at this K, so the model couldn't see it."));
    }
    rows.push(checkRow(r.ok, r.ok ? "Cited the source passage" : "Didn't cite the source passage", "This is the citation accuracy metric from the paper."));
  }
  if (!cited.length && !r.idk) rows.push(checkRow(false, "No citations", "The model didn't say where its answer came from."));
  for (const n of cited) {
    const st = STATUS[r.c[n]];
    if (r.c[n] === "x") {
      rows.push(checkRow(false, `Citation [${n}] points to nothing`, `Only ${k} document${k > 1 ? "s were" : " was"} given.`));
    } else if (isHallucinatedTrick(q, r) && r.c[n] !== "u") {
      rows.push(checkRow(null, `Citation [${n}] looks convincing`,
        "Its wording is close to the passage, but the passage doesn't answer this question. That's what makes this kind of hallucination hard to spot."));
    } else {
      rows.push(checkRow(st.ok, `Citation [${n}] ${st.label}`, "Sentence similarity check (approximate)"));
    }
  }
  return rows.join("");
}

// ── Passages ──────────────────────────────────────────────────────────
function renderPassages(q, r) {
  return r.d.map((docId, i) => {
    const n = String(i + 1);
    const cited = n in r.c;
    const text = state.passages[docId];
    const preview = text.length > 220 ? text.slice(0, 220) + "…" : text;
    const tags = [
      cited ? '<span class="tag tag-accent">cited</span>' : "",
      docId === q.doc ? '<span class="tag">source passage</span>' : "",
    ].join("");
    const more = text.length > 220 ? `<details><summary>Show full passage</summary><p>${esc(text)}</p></details>` : "";
    return `<div class="passage${cited ? " is-cited" : ""}"><div class="passage-head"><span class="num">[${n}]</span><span class="score">similarity ${r.s[i].toFixed(2)}</span>${tags}</div><p>${esc(preview)}</p>${more}</div>`;
  }).join("");
}

// ── Explore tab ───────────────────────────────────────────────────────
function renderExplore() {
  const q = current();
  const r = run();
  $("#output").innerHTML = `
    <div class="qcard">
      <div class="label">Question${q.ans ? "" : ' <span class="tag tag-accent">trick</span>'}</div>
      <h2>${esc(q.q)}</h2>
      ${q.en ? `<p class="muted">${esc(q.en)}</p>` : ""}
    </div>
    <div class="answer"><div class="label">Model answer · K = ${state.k}</div><p id="answer-text"></p></div>
    <div class="label section-label">Checks</div>
    <div>${renderChecks(q, r, state.k)}</div>
    <div class="label section-label">Retrieved documents (K = ${state.k})</div>
    <div>${renderPassages(q, r)}</div>`;
  renderAnswer(q, r, state.k);
  document.querySelectorAll("#k-picker button").forEach((b) => b.setAttribute("aria-checked", String(Number(b.dataset.k) === state.k)));
}

// ── Compare tab ───────────────────────────────────────────────────────
function renderCompare() {
  const q = current();
  const cards = K_VALUES.map((k) => {
    const r = run(q, k);
    let ok, verdict;
    if (q.ans) {
      ok = r.ok && !r.idk;
      verdict = ok ? "Cited the source" : r.idk ? "Said it doesn't know" : "Missed the source";
    } else {
      ok = r.idk;
      verdict = ok ? "Admitted it doesn't know" : "Hallucinated";
    }
    return `<div class="k-card ${ok ? "k-ok" : "k-bad"}"><div class="k-head"><span class="k">K = ${k}</span><span class="verdict">${ok ? "✓" : "✗"} ${verdict}</span></div><p>${citeChips(r.a, r, q, k)}</p></div>`;
  }).join("");
  $("#compare").innerHTML = `
    <p class="tab-intro">Same question, four values of K. Watch where answers change and where hallucinations appear. Pick another question in Explore and it shows up here.</p>
    <div class="qcard">
      <div class="label">Question${q.ans ? "" : ' <span class="tag tag-accent">trick</span>'}</div>
      <h2>${esc(q.q)}</h2>
      ${q.en ? `<p class="muted">${esc(q.en)}</p>` : ""}
      <p>${q.ans ? `Expected answer: <b>${esc(q.gold)}</b>` : "No document contains the answer."}</p>
    </div>
    <div class="k-grid">${cards}</div>
    <div class="chips compare-chips">${featuredChips()}</div>`;
}

// ── Findings tab ──────────────────────────────────────────────────────
function demoStats() {
  return K_VALUES.map((k) => {
    let ans = 0, ansOk = 0, trick = 0, trickIdk = 0;
    for (const q of state.questions) {
      const r = q.k[String(k)];
      if (q.ans) { ans++; if (r.ok) ansOk++; } else { trick++; if (r.idk) trickIdk++; }
    }
    return { k, cite: ansOk / ans, idk: trickIdk / trick };
  });
}

function renderFindings() {
  const w = 520, h = 240, pad = 36, bw = 70;
  const gap = (w - 2 * pad - bw * PAPER.length) / (PAPER.length - 1);
  const bars = PAPER.map((p, i) => {
    const x = pad + i * (bw + gap);
    const bh = (h - 70) * p.acc;
    const y = h - 34 - bh;
    return `<rect class="${p.k === 10 ? "bar-drop" : "bar"}" x="${x}" y="${y}" width="${bw}" height="${bh}" rx="8"/>
      <text class="bar-val" x="${x + bw / 2}" y="${y - 8}">${Math.round(p.acc * 100)}%</text>
      <text class="bar-k" x="${x + bw / 2}" y="${h - 12}">K = ${p.k}</text>`;
  }).join("");
  const paperRows = PAPER.map((p) => `<tr><td>${p.k}</td><td>${p.rougeL.toFixed(3)}</td><td>${p.bleu.toFixed(3)}</td><td>${p.bert.toFixed(3)}</td><td><b>${Math.round(p.acc * 100)}%</b></td></tr>`).join("");
  const demoRows = demoStats().map((s) => `<tr><td>${s.k}</td><td>${Math.round(s.cite * 100)}%</td><td>${Math.round(s.idk * 100)}%</td></tr>`).join("");
  const nAns = state.questions.filter((q) => q.ans).length;
  const nTrick = state.questions.length - nAns;

  $("#findings").innerHTML = `
    <div class="findings">
      <div class="big-numbers">
        <div><span class="big">76%</span><span>citation accuracy at K = 5, the best setting</span></div>
        <div><span class="big accent">36%</span><span>at K = 10: more documents, half the accuracy</span></div>
      </div>
      <div class="card"><div class="label">Citation accuracy by K (paper)</div>
        <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Citation accuracy by K: 67%, 70%, 76%, 36%">${bars}</svg></div>
      <div class="card takeaways"><div class="label">What we found</div><ol>
        <li><b>More context helps, up to a point.</b> Going from 1 to 5 documents made citations more accurate.</li>
        <li><b>Then it collapses.</b> At 10 documents the model loses track of where facts came from and cites the wrong one most of the time.</li>
        <li><b>Answer quality saturates too.</b> ROUGE, BLEU and BERTScore stay flat and then drop at K = 10, as irrelevant passages distract the model.</li>
      </ol></div>
      <div class="card"><div class="label">Paper results (100 questions per K)</div>
        <div class="table-wrap"><table><thead><tr><th>K</th><th>ROUGE-L</th><th>BLEU</th><th>BERTScore F1</th><th>Citation acc.</th></tr></thead><tbody>${paperRows}</tbody></table></div></div>
      <div class="card"><div class="label">This demo's run (all ${state.questions.length} questions)</div>
        <div class="table-wrap"><table><thead><tr><th>K</th><th>Cited the source (${nAns} answerable)</th><th>Admitted "don't know" (${nTrick} trick)</th></tr></thead><tbody>${demoRows}</tbody></table></div>
        <p class="muted small">Computed live in your browser from the answers on this site. The demo uses an improved prompt, so its numbers differ from the paper.</p></div>
      <div class="card credits"><div class="label">About this project</div>
        <p>Final assignment for <i>Kapita Selekta Sistem Cerdas</i> by <b>Aldi</b> (retrieval pipeline) and <b>Haiqal</b> (generation, evaluation and this demo).</p>
        <p>Dataset: <a href="https://huggingface.co/datasets/rifkiaputri/idk-mrc" target="_blank" rel="noopener">IDK-MRC</a> (Indonesian; half the questions have no answer). Search: multilingual MiniLM embeddings + FAISS. Model: Qwen2.5-1.5B-Instruct.</p>
        <p class="muted small">Every answer on this site was generated ahead of time by the model, for all ${state.questions.length} questions at each K, so pages load instantly. The demo uses Qwen's chat format and strips Wikipedia footnote marks like "[1]" from passages, so its answers are cleaner than in the paper.</p>
        <p><a href="https://github.com/mhaiqaldk/rag-saturasi-halusinasi-kssc" target="_blank" rel="noopener">Notebook, report and code on GitHub →</a></p></div>
    </div>`;
}

// ── Navigation, search and state ──────────────────────────────────────
function featuredChips() {
  return state.questions
    .map((q, i) => (q.en ? `<button type="button" class="chip" data-qi="${i}">${esc(q.en)}${q.ans ? "" : '<span class="chip-tag">trick</span>'}</button>` : ""))
    .join("");
}

function select(qi, { scroll = true } = {}) {
  state.qi = qi;
  $("#search").value = "";
  $("#results").hidden = true;
  syncHash();
  renderExplore();
  if (state.tab === "compare") renderCompare();
  if (scroll && window.innerWidth < 900 && state.tab === "explore") {
    $("#output").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }
}

function setK(k) {
  state.k = k;
  syncHash();
  renderExplore();
}

function showTab(name) {
  state.tab = name;
  for (const t of ["explore", "compare", "findings"]) {
    $(`#tab-${t}`).setAttribute("aria-selected", String(t === name));
    $(`#${t}`).hidden = t !== name;
  }
  if (name === "compare") renderCompare();
  if (name === "findings" && !$("#findings").innerHTML) renderFindings();
  syncHash();
}

function syncHash() {
  const params = new URLSearchParams({ q: String(state.qi), k: String(state.k) });
  if (state.tab !== "explore") params.set("tab", state.tab);
  history.replaceState(null, "", "#" + params.toString());
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const qi = Number(p.get("q"));
  const k = Number(p.get("k"));
  if (Number.isInteger(qi) && qi >= 0 && qi < state.questions.length && p.has("q")) state.qi = qi;
  if (K_VALUES.includes(k)) state.k = k;
  return p.get("tab");
}

function normalize(s) {
  return s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
}

function onSearch() {
  const term = normalize($("#search").value.trim());
  const box = $("#results");
  if (term.length < 2) { box.hidden = true; return; }
  const hits = [];
  state.questions.forEach((q, i) => {
    if (hits.length < 8 && (normalize(q.q).includes(term) || (q.en && normalize(q.en).includes(term)))) hits.push(i);
  });
  box.innerHTML = hits.length
    ? hits.map((i) => {
        const q = state.questions[i];
        return `<button type="button" role="option" class="result" data-qi="${i}">${esc(q.q)}${q.ans ? "" : '<span class="chip-tag">trick</span>'}</button>`;
      }).join("")
    : '<div class="no-results">No question matches. Try a word like "kapan", "siapa" or "Indonesia".</div>';
  box.hidden = false;
}

function bindEvents() {
  $("#k-picker").innerHTML = K_VALUES.map((k) => `<button type="button" role="radio" data-k="${k}" aria-checked="false">${k}</button>`).join("");
  $("#k-picker").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setK(Number(b.dataset.k)); });
  $("#featured").innerHTML = featuredChips();
  document.addEventListener("click", (e) => {
    const chip = e.target.closest("[data-qi]");
    if (chip) select(Number(chip.dataset.qi));
    if (!e.target.closest(".search-box")) $("#results").hidden = true;
  });
  $("#search").addEventListener("input", onSearch);
  $("#search").addEventListener("keydown", (e) => {
    if (e.key === "Escape") { $("#results").hidden = true; e.target.blur(); }
    if (e.key === "Enter") { const first = $("#results .result"); if (first) select(Number(first.dataset.qi)); }
  });
  $("#random").addEventListener("click", () => select(Math.floor(Math.random() * state.questions.length)));
  for (const t of ["explore", "compare", "findings"]) $(`#tab-${t}`).addEventListener("click", () => showTab(t));
}

async function init() {
  $("#output").innerHTML = '<div class="empty"><h3>Loading 3,000 answers…</h3><p>This takes a second.</p></div>';
  try {
    const [questions, passages] = await Promise.all([
      fetch("/data/questions.json").then((r) => r.json()),
      fetch("/data/passages.json").then((r) => r.json()),
    ]);
    state.questions = questions;
    state.passages = passages;
  } catch (err) {
    $("#output").innerHTML = '<div class="empty"><h3>Couldn\'t load the answers</h3><p>Check your connection and reload the page.</p></div>';
    return;
  }
  state.qi = Math.max(0, state.questions.findIndex((q) => q.q === DEFAULT_Q));
  const tab = readHash();
  bindEvents();
  renderExplore();
  if (tab === "compare" || tab === "findings") showTab(tab);
}

init();
