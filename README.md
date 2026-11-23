# Analisis Saturasi & Halusinasi pada RAG

How does the number of retrieved documents (Top-K) change answer quality and
**citation hallucination** in an Indonesian retrieval-augmented generation
(RAG) system?

Final assignment for *Kapita Selekta Sistem Cerdas* by **Aldi** (retrieval
pipeline) and **Haiqal** (generation, evaluation and the live demo).

**▶ Try it: [RAG Lab](https://rag-saturasi-halusinasi-kssc.vercel.app)**. Explore real answers to all 764
questions at four values of K and see where the model hallucinates.

## Results

100 questions from IDK-MRC per setting, Qwen2.5-1.5B-Instruct as the generator.

| K | ROUGE-L | BLEU | BERTScore F1 | Citation accuracy |
|---|---|---|---|---|
| 1 | 0.087 | 0.021 | 0.312 | 67% |
| 3 | 0.087 | 0.019 | 0.311 | 70% |
| 5 | 0.086 | 0.017 | 0.311 | **76%** |
| 10 | 0.074 | 0.017 | 0.297 | **36%** |

- **More context helps, up to a point:** citation accuracy rises from K = 1 to K = 5.
- **Then it collapses:** at K = 10 the model loses track of where facts came
  from and cites the wrong document most of the time.
- **Answer quality saturates:** ROUGE, BLEU and BERTScore stay flat, then drop at K = 10.

## How it works

1. **Search:** every passage in the IDK-MRC validation split (358 Indonesian
   Wikipedia passages) is embedded with `paraphrase-multilingual-MiniLM-L12-v2`
   and stored in a FAISS inner-product index. A question retrieves the top K passages.
2. **Answer:** Qwen2.5-1.5B-Instruct gets the numbered passages and must cite
   them as [1], [2]… or say "Tidak tahu" (don't know).
3. **Check:** ROUGE, BLEU, BERTScore against the gold answer, plus citation
   accuracy: does a citation point at the passage that actually holds the answer?

Half of IDK-MRC's questions have **no answer** in the text, which makes it a
good test of whether a model admits it doesn't know.

## Repository

| Path | What it is |
|---|---|
| `fixed_notebook(1).ipynb` | The full experiment (Colab, GPU) |
| `Final_Assignment_Kapita_Selekta_Aldi___Haiqal.pdf` | The report |
| `web/` | RAG Lab website: static HTML/JS over precomputed answers, deployed on Vercel |
| `demo/` | Live Gradio version that runs the model (CPU), plus the precompute scripts |

The website needs no server: `demo/precompute_all.py` ran the model on every
validation question at K = 1, 3, 5 and 10 and saved the answers, retrieved
documents and checks to `web/data/`. Preview it with
`python3 -m http.server -d web 5180`.

### Run the live demo locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r demo/requirements.txt
cd demo && ../.venv/bin/python app.py   # http://127.0.0.1:7860
```

The first run downloads the model (about 3 GB). `demo/build_data.py` and
`demo/precompute_compare.py` regenerate the sample questions and the
precomputed comparison answers.

The demo differs slightly from the notebook: it uses Qwen's chat format,
repeats the citation rule next to the question, and strips Wikipedia footnote
marks like "[1]" from passages (they looked exactly like citations). Its
answers are therefore cleaner than in the paper; the numbers above are from the paper.
