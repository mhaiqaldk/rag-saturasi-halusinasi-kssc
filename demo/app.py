"""RAG Lab: a live demo of "Analisis Saturasi & Halusinasi pada RAG".

Run locally:  python demo/app.py
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

import gradio as gr

from rag import K_VALUES, RagEngine

DATA = Path(__file__).parent / "data"
REPO = "https://github.com/mhaiqaldk/rag-saturasi-halusinasi-kssc"

SAMPLES = json.loads((DATA / "samples.json").read_text())
SAMPLE_BY_Q = {s["question"]: s for s in SAMPLES}
COMPARE = json.loads((DATA / "compare.json").read_text()) if (DATA / "compare.json").exists() else {}
with open(DATA / "paper_results.csv") as f:
    PAPER = list(csv.DictReader(f))

engine = RagEngine()

_CITE = re.compile(r"\[(\d+)\]")
STATUS_LABEL = {
    "supported": "matches the passage",
    "weak": "loosely matches",
    "unsupported": "doesn't match",
    "invented": "no such document",
}


def esc(text: str) -> str:
    return html.escape(text or "")


def sample_label(s: dict) -> str:
    return f"{s['english']}{'  · trick' if not s['answerable'] else ''}"


# ── Rendering ─────────────────────────────────────────────────────────
def render_answer(answer: str, k: int, support: dict[int, str] | None = None, done: bool = False) -> str:
    def chip(m: re.Match) -> str:
        n = int(m.group(1))
        status = (support or {}).get(n, "pending")
        if n > k:
            status = "invented"
        return f'<span class="cite cite-{status}">{n}</span>'

    body = _CITE.sub(chip, esc(answer)) or '<span class="muted">…</span>'
    cursor = "" if done else '<span class="cursor"></span>'
    return f'<div class="answer"><div class="label">Model answer</div><p>{body}{cursor}</p></div>'


def render_passages(docs: list[dict], cited: list[int], gold_doc_id: int | None) -> str:
    cards = []
    for i, d in enumerate(docs, start=1):
        classes = ["passage"]
        tags = []
        if i in cited:
            classes.append("is-cited")
            tags.append('<span class="tag tag-accent">cited</span>')
        if gold_doc_id is not None and d["doc_id"] == gold_doc_id:
            tags.append('<span class="tag">source passage</span>')
        text = esc(d["text"])
        preview = text[:220] + ("…" if len(text) > 220 else "")
        more = f"<details><summary>Show full passage</summary><p>{text}</p></details>" if len(text) > 220 else ""
        cards.append(
            f'<div class="{" ".join(classes)}">'
            f'<div class="passage-head"><span class="num">[{i}]</span>'
            f'<span class="score">similarity {d["score"]:.2f}</span>{"".join(tags)}</div>'
            f'<p>{preview}</p>{more}</div>'
        )
    return f'<div class="label">Retrieved documents (K = {len(docs)})</div>' + "".join(cards)


def check_row(ok: bool | None, title: str, detail: str = "") -> str:
    icon = {True: "✓", False: "✗", None: "•"}[ok]
    cls = {True: "ok", False: "bad", None: "info"}[ok]
    detail_html = f"<span>{detail}</span>" if detail else ""
    return f'<div class="check check-{cls}"><b>{icon}</b><div><strong>{title}</strong>{detail_html}</div></div>'


def render_checks(answer: str, docs: list[dict], sample: dict | None, support: list[dict]) -> str:
    rows = []
    cited = engine.cited_numbers(answer)
    unknown = engine.admitted_unknown(answer)

    if sample and not sample["answerable"]:
        rows.append(check_row(None, "Trick question", "The answer isn't in any of the 358 documents."))
        if unknown:
            rows.append(check_row(True, "Admitted it doesn't know", "This is the right answer here."))
        else:
            rows.append(check_row(False, "Hallucinated an answer", "It answered anyway, so whatever it said is made up."))
    elif sample:
        rows.append(check_row(None, "Expected answer", esc(sample["gold_answer"])))
        if unknown:
            rows.append(check_row(False, "Said it doesn't know", "The answer was in the documents."))
        correct = engine.citation_correct(answer, docs, sample["gold_doc_id"])
        retrieved = any(d["doc_id"] == sample["gold_doc_id"] for d in docs)
        if not retrieved:
            rows.append(check_row(False, "Source passage not retrieved", "Search didn't find it at this K, so the model couldn't see it."))
        rows.append(
            check_row(correct, "Cited the source passage" if correct else "Didn't cite the source passage",
                      "This is the citation accuracy metric from the paper.")
        )

    hallucinated_trick = bool(sample) and not sample["answerable"] and not unknown

    if not cited and not unknown:
        rows.append(check_row(False, "No citations", "The model didn't say where its answer came from."))
    for s in support:
        if s["status"] == "invented":
            rows.append(check_row(False, f"Citation [{s['n']}] points to nothing", f"Only {len(docs)} documents were given."))
        elif hallucinated_trick and s["status"] != "unsupported":
            rows.append(check_row(None, f"Citation [{s['n']}] looks convincing",
                                  "Its wording is close to the passage, but the passage doesn't answer this question. "
                                  "That's what makes this kind of hallucination hard to spot."))
        else:
            ok = {"supported": True, "unsupported": False}.get(s["status"])
            rows.append(check_row(ok, f"Citation [{s['n']}] {STATUS_LABEL[s['status']]}", f"Sentence similarity {s['score']:.2f} (approximate check)"))
    if not sample:
        rows.append(check_row(None, "Your own question", "There's no answer key, so only the citation checks apply."))
    return '<div class="label">Checks</div>' + "".join(rows)


# ── Ask tab ───────────────────────────────────────────────────────────
def ask(question: str, k: int):
    question = (question or "").strip()
    if not question:
        yield (render_empty(), "", "", "Type a question or pick one of the examples.")
        return
    question = question[:300]
    k = int(k)
    sample = SAMPLE_BY_Q.get(question)
    docs = engine.retrieve(question, k)
    gold = sample["gold_doc_id"] if sample else None
    passages = render_passages(docs, [], gold)
    yield (render_answer("", k), "", passages, f"Found {k} document{'s' if k > 1 else ''}. Writing the answer…")

    answer = ""
    for answer in engine.generate_stream(question, docs):
        yield (render_answer(answer, k), "", passages, "Writing the answer…")

    support = engine.support_check(answer, docs)
    support_map = {s["n"]: s["status"] for s in support}
    if sample and not sample["answerable"] and not engine.admitted_unknown(answer):
        # A hallucination on a trick question: never show its citations as green.
        support_map = {n: ("pending" if st == "supported" else st) for n, st in support_map.items()}
    cited = engine.cited_numbers(answer)
    yield (
        render_answer(answer, k, support_map, done=True),
        render_checks(answer, docs, sample, support),
        render_passages(docs, cited, gold),
        "Done. Try the same question with a different K.",
    )


def render_empty() -> str:
    return (
        '<div class="empty"><h3>Ask something, then change K</h3>'
        "<p>Pick an example below (trick questions have no answer in the documents) "
        "and watch whether the model admits it doesn't know.</p></div>"
    )


# ── Compare tab ───────────────────────────────────────────────────────
def render_compare(label: str) -> str:
    sample = next((s for s in SAMPLES if sample_label(s) == label), None)
    if not sample or sample["question"] not in COMPARE:
        return '<div class="empty"><p>Pick a question to compare.</p></div>'
    head = (
        f'<div class="compare-q"><div class="label">Question</div><h3>{esc(sample["question"])}</h3>'
        f'<p class="muted">{esc(sample["english"])}</p>'
        + (f'<p>Expected answer: <b>{esc(sample["gold_answer"])}</b></p>' if sample["answerable"]
           else '<p><span class="tag tag-accent">trick</span> No document contains the answer.</p>')
        + "</div>"
    )
    cards = []
    for row in COMPARE[sample["question"]]:
        if sample["answerable"]:
            ok = row["citation_correct"] and not row["admitted_unknown"]
            verdict = "Cited the source" if ok else ("Said it doesn't know" if row["admitted_unknown"] else "Missed the source")
        else:
            ok = row["admitted_unknown"]
            verdict = "Admitted it doesn't know" if ok else "Hallucinated"
        body = _CITE.sub(lambda m: f'<span class="cite cite-pending">{m.group(1)}</span>', esc(row["answer"]))
        cards.append(
            f'<div class="k-card {"k-ok" if ok else "k-bad"}"><div class="k-head"><span class="k">K = {row["k"]}</span>'
            f'<span class="verdict">{"✓" if ok else "✗"} {verdict}</span></div><p>{body}</p></div>'
        )
    note = '<p class="muted small">Answers were generated ahead of time with the same model and prompt, so this tab loads instantly.</p>'
    return head + f'<div class="k-grid">{"".join(cards)}</div>' + note


# ── Findings tab ──────────────────────────────────────────────────────
def findings_html() -> str:
    acc = [(int(r["K"]), float(r["Citation Accuracy"])) for r in PAPER]
    w, h, pad, bw = 520, 240, 36, 70
    gap = (w - 2 * pad - bw * len(acc)) / (len(acc) - 1)
    bars = []
    for i, (k, v) in enumerate(acc):
        x = pad + i * (bw + gap)
        bh = (h - 70) * v
        y = h - 34 - bh
        cls = "bar-drop" if k == 10 else "bar"
        bars.append(
            f'<rect class="{cls}" x="{x:.0f}" y="{y:.0f}" width="{bw}" height="{bh:.0f}" rx="8"/>'
            f'<text class="bar-val" x="{x + bw / 2:.0f}" y="{y - 8:.0f}">{v:.0%}</text>'
            f'<text class="bar-k" x="{x + bw / 2:.0f}" y="{h - 12}">K = {k}</text>'
        )
    chart = f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Citation accuracy by K: 67%, 70%, 76%, 36%">{"".join(bars)}</svg>'

    rows = "".join(
        f"<tr><td>{r['K']}</td><td>{float(r['ROUGE-L']):.3f}</td><td>{float(r['BLEU']):.3f}</td>"
        f"<td>{float(r['BERTScore F1']):.3f}</td><td><b>{float(r['Citation Accuracy']):.0%}</b></td></tr>"
        for r in PAPER
    )
    return f"""
<div class="findings">
  <div class="big-numbers">
    <div><span class="big">76%</span><span>citation accuracy at K = 5, the best setting</span></div>
    <div><span class="big accent">36%</span><span>at K = 10: more documents, half the accuracy</span></div>
  </div>
  <div class="card"><div class="label">Citation accuracy by K</div>{chart}</div>
  <div class="card takeaways">
    <div class="label">What we found</div>
    <ol>
      <li><b>More context helps, up to a point.</b> Going from 1 to 5 documents made citations more accurate.</li>
      <li><b>Then it collapses.</b> At 10 documents the model loses track of where facts came from and cites the wrong one most of the time.</li>
      <li><b>Answer quality saturates too.</b> ROUGE, BLEU and BERTScore stay flat and then drop at K = 10, as irrelevant passages distract the model.</li>
    </ol>
  </div>
  <div class="card">
    <div class="label">All metrics (100 questions per K)</div>
    <div class="table-wrap"><table>
      <thead><tr><th>K</th><th>ROUGE-L</th><th>BLEU</th><th>BERTScore F1</th><th>Citation acc.</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
  </div>
  <div class="card credits">
    <div class="label">About this project</div>
    <p>Final assignment for <i>Kapita Selekta Sistem Cerdas</i> by <b>Aldi</b> (retrieval pipeline) and <b>Haiqal</b> (generation, evaluation and this demo).</p>
    <p>Dataset: <a href="https://huggingface.co/datasets/rifkiaputri/idk-mrc" target="_blank" rel="noopener">IDK-MRC</a> (Indonesian, half the questions have no answer).
    Search: multilingual MiniLM embeddings + FAISS. Model: Qwen2.5-1.5B-Instruct.</p>
    <p class="muted small">The live demo uses Qwen's chat format and strips Wikipedia footnote marks like "[1]" from passages, so its answers are cleaner than in the paper. The numbers above are from the paper.</p>
    <p><a href="{REPO}" target="_blank" rel="noopener">Notebook and report on GitHub →</a></p>
  </div>
</div>"""


HEADER = f"""
<header class="hero">
  <div class="glow"></div>
  <span class="eyebrow">RAG Lab · live research demo</span>
  <h1>Watch an AI make things&nbsp;up</h1>
  <p>Ask a question, choose how many documents the model reads (K), and see when it cites its sources correctly and when it hallucinates.</p>
  <a class="repo-link" href="{REPO}" target="_blank" rel="noopener">Read the research ↗</a>
</header>"""

CSS = (Path(__file__).parent / "style.css").read_text()

theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fff1eb", c100="#ffe0d1", c200="#ffbfa3", c300="#ff9a70", c400="#ff7a45",
        c500="#ff5a1f", c600="#e64a12", c700="#bf3b0d", c800="#99300d", c900="#7a280e", c950="#421206",
    ),
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0F0E13",
    body_background_fill_dark="#0F0E13",
    body_text_color="#F6F2EA",
    body_text_color_dark="#F6F2EA",
    body_text_color_subdued="#A8A29E",
    body_text_color_subdued_dark="#A8A29E",
    block_background_fill="#17161D",
    block_background_fill_dark="#17161D",
    block_border_color="#2A2833",
    block_border_color_dark="#2A2833",
    block_label_text_color="#A8A29E",
    block_label_text_color_dark="#A8A29E",
    block_title_text_color="#F6F2EA",
    block_title_text_color_dark="#F6F2EA",
    input_background_fill="#1F1D26",
    input_background_fill_dark="#1F1D26",
    input_border_color="#34313E",
    input_border_color_dark="#34313E",
    border_color_primary="#34313E",
    border_color_primary_dark="#34313E",
    button_primary_background_fill="#FF5A1F",
    button_primary_background_fill_dark="#FF5A1F",
    button_primary_background_fill_hover="#FF7A45",
    button_primary_background_fill_hover_dark="#FF7A45",
    button_primary_text_color="#0F0E13",
    button_primary_text_color_dark="#0F0E13",
    button_secondary_background_fill="#1F1D26",
    button_secondary_background_fill_dark="#1F1D26",
    button_secondary_text_color="#F6F2EA",
    button_secondary_text_color_dark="#F6F2EA",
    color_accent_soft="#2A1A14",
    color_accent_soft_dark="#2A1A14",
    checkbox_label_background_fill_selected="#FF5A1F",
    checkbox_label_background_fill_selected_dark="#FF5A1F",
    checkbox_label_text_color_selected="#0F0E13",
    checkbox_label_text_color_selected_dark="#0F0E13",
)

with gr.Blocks(title="RAG Lab · when does an AI make things up?", css=CSS, theme=theme) as demo:
    gr.HTML(HEADER)
    with gr.Tabs():
        with gr.Tab("Ask"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=300):
                    question = gr.Textbox(
                        label="Question (Indonesian works best)",
                        placeholder="e.g. Apa kepanjangan dari GPS?",
                        max_length=300,
                        lines=2,
                    )
                    k = gr.Radio(K_VALUES, value=3, label="Documents to read (K)",
                                 info="Low K: may miss the source. High K: more noise.")
                    ask_btn = gr.Button("Ask the model", variant="primary", size="lg")
                    status = gr.Markdown("", elem_classes="status")
                    gr.Examples(
                        examples=[[s["question"]] for s in SAMPLES],
                        inputs=[question],
                        example_labels=[sample_label(s) for s in SAMPLES],
                        label="Examples from the dataset",
                        examples_per_page=12,
                    )
                with gr.Column(scale=7, min_width=300):
                    answer_out = gr.HTML(render_empty())
                    checks_out = gr.HTML()
                    passages_out = gr.HTML()
            outputs = [answer_out, checks_out, passages_out, status]
            ask_btn.click(ask, [question, k], outputs, scroll_to_output=True)
            question.submit(ask, [question, k], outputs, scroll_to_output=True)

        with gr.Tab("Compare K"):
            gr.Markdown("Same question, four values of K. Watch where answers change and where hallucinations appear.",
                        elem_classes="tab-intro")
            pick = gr.Dropdown([sample_label(s) for s in SAMPLES], value=sample_label(SAMPLES[6]),
                               label="Question", interactive=True)
            compare_out = gr.HTML(render_compare(sample_label(SAMPLES[6])))
            pick.change(render_compare, pick, compare_out)

        with gr.Tab("Findings"):
            gr.HTML(findings_html())

    gr.HTML('<footer class="foot">Built by Haiqal &amp; Aldi</footer>')

demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch()
