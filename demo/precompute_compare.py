"""One-off: answer every sample at K = 1, 3, 5, 10 and save data/compare.json,
so the Compare tab loads instantly. Run after build_data.py:
    python demo/precompute_compare.py
"""

import json
from pathlib import Path

from rag import K_VALUES, RagEngine

DATA = Path(__file__).parent / "data"


def main():
    engine = RagEngine()
    samples = json.loads((DATA / "samples.json").read_text())
    out = {}
    for s in samples:
        rows = []
        for k in K_VALUES:
            docs = engine.retrieve(s["question"], k)
            answer = engine.generate(s["question"], docs)
            rows.append(
                {
                    "k": k,
                    "answer": answer,
                    "doc_ids": [d["doc_id"] for d in docs],
                    "cited": engine.cited_numbers(answer),
                    "citation_correct": engine.citation_correct(answer, docs, s["gold_doc_id"]),
                    "admitted_unknown": engine.admitted_unknown(answer),
                }
            )
            print(f"K={k:<2} {s['question'][:40]:<40} {answer[:70]!r}")
        out[s["question"]] = rows
    (DATA / "compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved {len(out)} questions to {DATA / 'compare.json'}")


if __name__ == "__main__":
    main()
