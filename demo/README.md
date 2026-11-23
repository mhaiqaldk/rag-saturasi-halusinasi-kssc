---
title: RAG Lab
emoji: 🔎
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Watch an AI cite sources, or make things up, as K changes
---

# RAG Lab

Live demo of **"Analisis Saturasi & Halusinasi pada RAG"**: how the number of
retrieved documents (Top-K) changes answer quality and citation hallucination
in an Indonesian RAG system.

- **Ask:** type a question or pick an example, choose K, and see the answer,
  its citations, the retrieved documents and automatic checks.
- **Compare K:** the same question answered at K = 1, 3, 5 and 10.
- **Findings:** the results from the paper.

Stack: IDK-MRC dataset, multilingual MiniLM embeddings + FAISS for search,
Qwen2.5-1.5B-Instruct on CPU for answers, Gradio for the interface.

Research and code: https://github.com/mhaiqaldk/rag-saturasi-halusinasi-kssc
By Aldi (retrieval) and Haiqal (generation, evaluation and this demo).
