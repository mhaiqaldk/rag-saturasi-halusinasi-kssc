# RAG Lab: live version (Gradio)

The online site (`web/`) shows answers the model generated ahead of time.
This folder runs the model **live**, so you can type your own questions.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r demo/requirements.txt
cd demo && ../.venv/bin/python app.py   # http://127.0.0.1:7860
```

The first run downloads Qwen2.5-1.5B-Instruct (about 3 GB). Answers take a few
seconds on a laptop CPU.

| File | What it does |
|---|---|
| `rag.py` | Retrieval, generation and citation checks, ported from the notebook |
| `app.py`, `style.css` | The Gradio interface |
| `build_data.py` | Writes the 12 featured sample questions to `data/samples.json` |
| `precompute_compare.py` | Answers the samples at every K for the Compare tab |
| `precompute_all.py` | Answers all 764 questions at every K for the static site (`web/data/`) |

On macOS, `torch` must be imported before `faiss`, or Python crashes; `rag.py`
already does this.
