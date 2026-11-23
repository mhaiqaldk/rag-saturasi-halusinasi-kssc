"""One-off: write data/samples.json with the demo's sample questions.

Each sample is a real IDK-MRC validation question. "Trick" samples ask about
the same topic as a real passage but their answer is not in any document.
Run:  python demo/build_data.py
"""

import json
from pathlib import Path

from datasets import load_dataset

from rag import RagEngine

# (question exactly as in IDK-MRC, English translation)
SAMPLES = [
    ("Apa kepanjangan dari GPS?", "What does GPS stand for?"),
    ("Apakah jenis hiu terkecil di dunia?", "What is the smallest shark in the world?"),
    ("Apakah nama lagu kebangsaan Jepang?", "What is Japan's national anthem called?"),
    ("Kapan telepon pertama kali digunakan di Indonesia ?", "When was the telephone first used in Indonesia?"),
    ("Kapan telepon pertama kali digunakan di China?", "When was the telephone first used in China?"),
    ("apakah nama mata uang Taiwan ?", "What is Taiwan's currency called?"),
    ("Kapan PDI dibubarkan?", "When was the PDI party dissolved?"),
    ("Kapan Walt Disney Animation Studios berdiri ?", "When was Walt Disney Animation Studios founded?"),
    ('Siapakah nama suami Sir Alexander Chapman "Alex" Ferguson?', "What is the name of Sir Alex Ferguson's husband?"),
    ("dari manakah tari angin mamiri berasal?", "Where does the Angin Mamiri dance come from?"),
    ("Kapan Toko Gunung Agung ditutup?", "When did the Gunung Agung bookstore close?"),
    ("Apakah jenis hiu terbesar di dunia ?", "What is the largest shark in the world?"),
]


def main():
    val = load_dataset("rifkiaputri/idk-mrc")["validation"]
    engine = RagEngine(load_generator=False)

    lookup = {}
    for item in val:
        for qa in item["qas"]:
            lookup[qa["question"].strip()] = (qa, item["context"])

    out = []
    for question, english in SAMPLES:
        qa, ctx = lookup[question.strip()]
        out.append(
            {
                "question": question.strip(),
                "english": english,
                "answerable": not qa["is_impossible"],
                "gold_answer": qa["answers"][0]["text"] if qa["answers"] else None,
                "gold_doc_id": engine.context_to_docid[ctx],
            }
        )

    path = Path(__file__).parent / "data" / "samples.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {len(out)} samples to {path}")


if __name__ == "__main__":
    main()
