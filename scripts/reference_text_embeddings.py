"""Reference text embeddings from the official sentence-transformers implementation.

Writes tests/fixtures/reference_text.json, which tests/parity.rs compares the
Rust/ONNX path against (token ids and cosine similarity).

    pip install "sentence-transformers>=6.1" "transformers>=5.19" torch
    python scripts/reference_text_embeddings.py tests/fixtures/reference_text.json

Runs in float32: the model card warns that float16 overflows.
"""
import json, sys, torch
from sentence_transformers import SentenceTransformer
out = sys.argv[1]
m = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"dtype": torch.float32},
                        config_kwargs={"vision_config": None, "audio_config": None})
queries = ["tomato soup recipe", "how does redis caching work"]
docs = [(None, "Simmer tomatoes with garlic and basil for twenty minutes, then blend."),
        ("Redis", "Redis keeps hot keys in memory so repeated reads avoid the database.")]
tok = m.tokenizer
res = {"queries": [], "docs": []}
for q in queries:
    s = "task: search result | query: " + q
    e = m.encode(q, prompt_name="SearchQuery", normalize_embeddings=True)
    res["queries"].append({"text": q, "input": s, "ids": tok(s)["input_ids"], "emb": e.tolist()})
for t, d in docs:
    s = f"title: {t or 'none'} | text: {d}"
    e = m.encode(s, normalize_embeddings=True)
    res["docs"].append({"title": t, "text": d, "input": s, "ids": tok(s)["input_ids"], "emb": e.tolist()})
json.dump(res, open(out, "w"))
import numpy as np
Q = np.array([x["emb"] for x in res["queries"]]); D = np.array([x["emb"] for x in res["docs"]])
print(np.round(Q @ D.T, 3)); print("ids0", res["queries"][0]["ids"][:12], len(res["queries"][0]["ids"]))
