#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive query tool:
- Loads a FAISS index (128/256/512/768D).
- Lets you type questions.
- Encodes them with EmbeddingGemma (768D), slices down to index dim, normalizes, retrieves top-k.
- Prints top results with park_code, title, content_type, and url.

Usage:
  python scripts/query_demo.py --index artifacts/faiss/parks_flatip_256.index \
                               --meta artifacts/embeddings/meta.jsonl \
                               --model google/embeddinggemma-300m \
                               --k 5
"""

import argparse, pathlib, json, numpy as np, faiss, torch
from sentence_transformers import SentenceTransformer

def l2n(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def infer_park_code(q: str) -> str | None:
    ql = q.lower()
    if "yosemite" in ql or "tioga" in ql or "half dome" in ql:
        return "yose"
    if "grand canyon" in ql or "south rim" in ql or "north rim" in ql or "phantom ranch" in ql:
        return "grca"
    if "sequoia" in ql or "kings canyon" in ql or "general sherman" in ql or "grant grove" in ql:
        return "seki"
    return None

POLICY_KEYWORDS = [
    "reservation", "reservations", "permit", "permits",
    "backcountry", "lottery", "campground", "camping",
    "fees", "passes", "entrance", "hours", "seasonal", "open year-round",
    "shuttle", "parking", "road closure", "tioga road", "opening", "closing"
]

def looks_like_policy(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in POLICY_KEYWORDS)

def main():
    ap = argparse.ArgumentParser(description="Policy-aware RAG query")
    ap.add_argument("--index", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", default="google/embeddinggemma-300m")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--probe", type=int, default=50, help="initial topK before filters")
    args = ap.parse_args()

    # Load FAISS + meta
    idx = faiss.read_index(args.index)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]
    D = idx.d

    # Load embedder
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    def l2n(a):
        import numpy as np
        n = (a**2).sum(axis=1, keepdims=True)**0.5 + 1e-12
        return a / n

    while True:
        q = input("\nEnter a question (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break

        park_pref = infer_park_code(q)
        policy_intent = looks_like_policy(q)

        # Encode
        q768 = model.encode([q], convert_to_numpy=True, normalize_embeddings=False)
        qvec = l2n(q768[:, :D].astype(np.float32))

        # Broad search first
        scores, ids = idx.search(qvec, max(args.probe, args.k))
        cand = []
        for sc, ii in zip(scores[0], ids[0]):
            m = meta[int(ii)]
            cand.append((float(sc), int(ii), m))

        # Soft filters/re-rank
        # 1) Prefer inferred park
        if park_pref:
            for i,(sc, ii, m) in enumerate(cand):
                if m.get("park_code") == park_pref:
                    cand[i] = (sc + 0.10, ii, m)  # small boost

        # 2) If policy intent, deprioritize 'places' a bit, boost likely policy types
        if policy_intent:
            for i,(sc, ii, m) in enumerate(cand):
                ct = (m.get("content_type") or "").lower()
                if ct == "places":
                    cand[i] = (sc - 0.15, ii, m)  # slight penalty
                elif ct in {"alerts", "parks", "campgrounds", "feespasses", "hours", "articles", "events", "thingstodo"}:
                    cand[i] = (sc + 0.05, ii, m)

        # Final rank
        cand.sort(key=lambda x: x[0], reverse=True)
        top = cand[:args.k]

        print(f"\nTop {args.k} results for: {q}")
        for rank, (sc, ii, m) in enumerate(top, start=1):
            print(f"{rank:>2}. score={sc:.4f}  [{m.get('park_code')}] {m.get('title','')} ({m.get('content_type')})")
            print(f"    {m.get('url')}")
        print("-"*60)

if __name__ == "__main__":
    main()
