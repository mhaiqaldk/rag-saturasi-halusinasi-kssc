"""Answer every IDK-MRC validation question at K = 1, 3, 5, 10 and write the
data files for the static site in web/data/:

  passages.json   the 358 cleaned passages (index = doc_id)
  questions.json  every question with its answers, citations and checks

Run (takes about an hour on an Apple Silicon Mac):
    python demo/precompute_all.py
Resumes from web/data/.progress.json if interrupted.
"""

import json
import time
from pathlib import Path

from datasets import load_dataset

from rag import K_VALUES, RagEngine

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "data"
SAMPLES = json.loads((Path(__file__).parent / "data" / "samples.json").read_text())
ENGLISH = {s["question"]: s["english"] for s in SAMPLES}
STATUS_CODE = {"supported": "s", "weak": "w", "unsupported": "u", "invented": "x"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    engine = RagEngine()
    val = load_dataset("rifkiaputri/idk-mrc")["validation"]

    (OUT / "passages.json").write_text(
        json.dumps([d["text"] for d in engine.corpus], ensure_ascii=False, separators=(",", ":"))
    )

    items = []
    for row in val:
        for qa in row["qas"]:
            items.append((qa, engine.context_to_docid[row["context"]]))

    progress_path = OUT / ".progress.json"
    done = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    start = time.time()

    for n, (qa, gold) in enumerate(items, start=1):
        qid = qa["id"]
        if qid in done:
            continue
        question = qa["question"].strip()
        runs = {}
        for k in K_VALUES:
            docs = engine.retrieve(question, k)
            answer = engine.generate(question, docs)
            support = engine.support_check(answer, docs)
            runs[str(k)] = {
                "d": [d["doc_id"] for d in docs],
                "s": [round(d["score"], 2) for d in docs],
                "a": answer,
                "c": {str(x["n"]): STATUS_CODE[x["status"]] for x in support},
                "ok": engine.citation_correct(answer, docs, gold),
                "idk": engine.admitted_unknown(answer),
            }
        entry = {
            "q": question,
            "ans": not qa["is_impossible"],
            "gold": qa["answers"][0]["text"] if qa["answers"] else None,
            "doc": gold,
            "k": runs,
        }
        if question in ENGLISH:
            entry["en"] = ENGLISH[question]
        done[qid] = entry
        if n % 10 == 0 or n == len(items):
            progress_path.write_text(json.dumps(done, ensure_ascii=False))
            rate = (time.time() - start) / max(1, len(done))
            print(f"{len(done)}/{len(items)} questions, ~{rate * (len(items) - len(done)) / 60:.0f} min left", flush=True)

    ordered = [done[qa["id"]] for qa, _ in items]
    (OUT / "questions.json").write_text(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")))
    progress_path.unlink(missing_ok=True)
    print(f"DONE: wrote {len(ordered)} questions to {OUT}", flush=True)


if __name__ == "__main__":
    main()
