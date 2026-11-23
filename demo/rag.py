"""Core RAG logic for the live demo, ported from fixed_notebook(1).ipynb.

Retrieval (Aldi): multilingual MiniLM embeddings + FAISS inner-product index
over the unique passages of the IDK-MRC validation split.
Generation and checks (Haiqal): Qwen2.5-1.5B-Instruct with a citation prompt,
plus the citation checks used in the paper.
"""

from __future__ import annotations

import os
import re
from threading import Thread
from typing import Iterator

import numpy as np
import torch  # must be imported before faiss: on macOS the reverse order segfaults (OpenMP clash)
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import faiss

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
K_VALUES = [1, 3, 5, 10]
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "150"))

# Wikipedia footnote marks such as "[1]" look exactly like citation marks,
# which confused citation parsing in the notebook, so passages are cleaned.
_FOOTNOTE = re.compile(r"\s*\[\d+\]")
_CITATION = re.compile(r"\[(\d+)\]")
_UNKNOWN = re.compile(r"tidak\s+tahu|tidak\s+(ada|disebutkan|terdapat)\s+(dalam|di)\s+konteks", re.I)

SYSTEM_PROMPT = (
    "Kamu adalah asisten yang menjawab pertanyaan berdasarkan konteks yang diberikan.\n"
    "Jawablah dalam Bahasa Indonesia secara singkat dan tepat.\n"
    "Sertakan nomor referensi [1], [2], dst. di dalam jawabanmu sesuai dokumen yang kamu gunakan.\n"
    'Jika jawabannya tidak ada dalam konteks, katakan "Tidak tahu".'
)

# With the chat template, Qwen ignored the citation rule when it only lived in
# the system message, so it is repeated next to the question with an example.
ANSWER_RULES = (
    "Jawab singkat dalam satu atau dua kalimat. Setiap kalimat WAJIB diakhiri nomor "
    'dokumen sumbernya, contoh: "Ibu kota Indonesia adalah Jakarta [2]." '
    'Jika jawabannya tidak ada dalam konteks, jawab hanya: "Tidak tahu."'
)


def pick_device() -> str:
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_passage(text: str) -> str:
    return _FOOTNOTE.sub("", text).strip()


class RagEngine:
    def __init__(self, load_generator: bool = True):
        self.device = pick_device()
        val = load_dataset("rifkiaputri/idk-mrc")["validation"]

        self.corpus: list[dict] = []
        self.context_to_docid: dict[str, int] = {}
        for item in val:
            ctx = item["context"]
            if ctx not in self.context_to_docid:
                self.context_to_docid[ctx] = len(self.corpus)
                self.corpus.append({"doc_id": len(self.corpus), "text": clean_passage(ctx)})

        self.embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
        emb = self.embedder.encode(
            [d["text"] for d in self.corpus], batch_size=64, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

        self.tokenizer = None
        self.model = None
        if load_generator:
            self.tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME, torch_dtype=dtype)
            self.model.to(self.device).eval()

    # ── Retrieval (notebook cell 12) ───────────────────────────────────
    def retrieve(self, query: str, k: int) -> list[dict]:
        q = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, k)
        return [
            {**self.corpus[i], "score": float(s)} for s, i in zip(scores[0], idx[0])
        ]

    @staticmethod
    def build_messages(query: str, docs: list[dict]) -> list[dict]:
        context = "\n\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs, start=1))
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Konteks:\n{context}\n\nPertanyaan: {query}\n\n{ANSWER_RULES}"},
        ]

    # ── Generation (notebook cell 17, now with the chat template) ──────
    def generate_stream(self, query: str, docs: list[dict]) -> Iterator[str]:
        """Yields the answer so far, token by token."""
        prompt = self.tokenizer.apply_chat_template(
            self.build_messages(query, docs), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = Thread(target=self._generate, kwargs=kwargs, daemon=True)
        thread.start()
        text = ""
        for piece in streamer:
            text += piece
            yield text
        thread.join()

    def _generate(self, **kwargs):
        with torch.no_grad():
            self.model.generate(**kwargs)

    def generate(self, query: str, docs: list[dict]) -> str:
        answer = ""
        for answer in self.generate_stream(query, docs):
            pass
        return answer.strip()

    # ── Checks (notebook cell 19, plus a support check) ────────────────
    @staticmethod
    def cited_numbers(answer: str) -> list[int]:
        """1-based citation numbers in order of first appearance."""
        seen: list[int] = []
        for n in _CITATION.findall(answer):
            if int(n) not in seen:
                seen.append(int(n))
        return seen

    @staticmethod
    def admitted_unknown(answer: str) -> bool:
        return bool(_UNKNOWN.search(answer))

    def citation_correct(self, answer: str, docs: list[dict], gold_doc_id: int) -> bool:
        """Paper metric: does any citation point at the gold passage?"""
        for n in self.cited_numbers(answer):
            if 1 <= n <= len(docs) and docs[n - 1]["doc_id"] == gold_doc_id:
                return True
        return False

    def support_check(self, answer: str, docs: list[dict]) -> list[dict]:
        """For each cited number: does the citing sentence resemble the passage?

        Approximate: best cosine similarity between the citing sentence and any
        sentence of the cited passage. Numbers beyond K are invented citations.
        """
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
        results = []
        for n in self.cited_numbers(answer):
            if not 1 <= n <= len(docs):
                results.append({"n": n, "status": "invented", "score": None})
                continue
            claim = " ".join(s for s in sentences if f"[{n}]" in s)
            claim = _CITATION.sub("", claim).strip()
            if not claim:
                results.append({"n": n, "status": "weak", "score": 0.0})
                continue
            passage_sents = [s for s in re.split(r"(?<=[.!?])\s+", docs[n - 1]["text"]) if s.strip()]
            emb = self.embedder.encode([claim] + passage_sents, convert_to_numpy=True, normalize_embeddings=True)
            score = float(np.max(emb[1:] @ emb[0]))
            status = "supported" if score >= 0.6 else "weak" if score >= 0.4 else "unsupported"
            results.append({"n": n, "status": status, "score": round(score, 2)})
        return results
