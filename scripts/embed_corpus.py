#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embed the parks corpus with google/embeddinggemma-300m (768-D).

Inputs:
  data/clean/parks_corpus.jsonl   # one JSON per line with "body_text" and metadata

Outputs:
  artifacts/embeddings/parks_768_fp16.npy   # [N, 768], float16, L2-normalized
  artifacts/embeddings/meta.jsonl           # aligned metadata {id, url, title, park_code, content_type}
  artifacts/reports/latency_embed.csv       # batch timings (ms/text, tokens/sec)
  artifacts/embeddings/env.txt              # environment & device info

Usage:
  python scripts/embed_corpus.py \
    --input data/clean/parks_corpus.jsonl \
    --model google/embeddinggemma-300m \
    --batch-size 16 \
    --out-npy artifacts/embeddings/parks_768_fp16.npy \
    --out-meta artifacts/embeddings/meta.jsonl
"""

import os
import json
import time
import argparse
import pathlib
from typing import List, Dict, Any

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer, util


def ensure_dir(p: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]):
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def now_ms() -> float:
    return time.time() * 1000.0


def embed_corpus(
    input_jsonl: pathlib.Path,
    model_name: str,
    batch_size: int,
    out_npy: pathlib.Path,
    out_meta: pathlib.Path,
    out_latency_csv: pathlib.Path,
    device: str = None,
    precision: str = "fp16",
):
    # Load data
    rows = read_jsonl(input_jsonl)
    texts = [r.get("body_text", "") for r in rows]
    if not texts:
        raise SystemExit(f"No texts found in {input_jsonl}")

    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"[Model] Loading {model_name} on {device} ...")
    model = SentenceTransformer(model_name, device=device)

    # Optional: force max seq len if needed (you said ST wrapper shows 2048)
    # model.max_seq_length = 2048

    # Precision
    if device == "cuda" and precision.lower() in ("fp16", "half"):
        model = model.half()
        torch.set_default_dtype(torch.float16)
        dtype_out = np.float16
    else:
        dtype_out = np.float32

    # Embed in batches
    all_vecs = []
    meta_rows = []
    lat_records = []

    total = len(texts)
    print(f"[Data] {total} chunks to embed; batch={batch_size}")
    start_all = now_ms()

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        t0 = now_ms()
        with torch.inference_mode():
            # Sentence-Transformers returns numpy when convert_to_numpy=True
            vecs = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                convert_to_numpy=True,
                normalize_embeddings=False,  # we'll normalize explicitly
                show_progress_bar=False,
            )
        t1 = now_ms()

        # Convert dtype and normalize
        if vecs.dtype != np.float32 and vecs.dtype != np.float16:
            vecs = vecs.astype(np.float32, copy=False)
        if dtype_out == np.float16 and vecs.dtype != np.float16:
            vecs = vecs.astype(np.float16)

        # Normalize to unit length (Matryoshka slices expect cosine-safe scale)
        # Cast to float32 for stable normalization math, then back to desired dtype
        v32 = vecs.astype(np.float32, copy=False)
        v32 = l2_normalize(v32)
        vecs = v32.astype(dtype_out, copy=False)

        all_vecs.append(vecs)

        # Metadata alignment (one meta row per vector)
        for j, r in enumerate(rows[i : i + batch_size]):
            meta_rows.append(
                {
                    "id": i + j,
                    "park_code": r.get("park_code", ""),
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content_type": r.get("content_type", ""),
                }
            )

        # Batch timing
        elapsed_ms = t1 - t0
        ms_per_text = elapsed_ms / max(1, len(batch_texts))
        lat_records.append(
            {
                "start_index": i,
                "batch_size": len(batch_texts),
                "elapsed_ms": round(elapsed_ms, 3),
                "ms_per_text": round(ms_per_text, 3),
            }
        )
        if (i // batch_size) % 10 == 0:
            print(f"  [Batch {i}-{i+len(batch_texts)}] {ms_per_text:.2f} ms/text")

    end_all = now_ms()
    wall_ms = end_all - start_all
    print(f"[Done] Embedding time: {wall_ms/1000:.2f}s for {total} texts")

    # Concatenate and save
    ensure_dir(out_npy)
    vec_mat = np.concatenate(all_vecs, axis=0)
    np.save(out_npy, vec_mat)
    print(f"[Save] Vectors: {vec_mat.shape} → {out_npy}")

    write_jsonl(out_meta, meta_rows)
    print(f"[Save] Meta rows: {len(meta_rows)} → {out_meta}")

    # Save latency CSV
    ensure_dir(out_latency_csv)
    import csv
    with out_latency_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["start_index", "batch_size", "elapsed_ms", "ms_per_text"])
        w.writeheader()
        for rec in lat_records:
            w.writerow(rec)
    print(f"[Save] Latency records: {len(lat_records)} → {out_latency_csv}")

    # Environment capture
    env_txt = out_npy.parent / "env.txt"
    ensure_dir(env_txt)
    with env_txt.open("w", encoding="utf-8") as f:
        f.write(f"device={device}\n")
        f.write(f"precision={precision}\n")
        f.write("cpu_num=NA\n")
        f.write(f"ram_GB={round(psutil.virtual_memory().total/ (1024**3), 2)}\n")
        try:
            import transformers, sentence_transformers
            f.write(f"transformers={transformers.__version__}\n")
            f.write(f"sentence_transformers={sentence_transformers.__version__}\n")
            f.write(f"torch={torch.__version__}\n")
        except Exception:
            pass
        try:
            # Model hash/revision if available
            rev = getattr(model, "model_card", None)
            if rev:
                f.write(f"model_card={str(rev)}\n")
        except Exception:
            pass
    print(f"[Save] Env info → {env_txt}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=pathlib.Path, help="JSONL corpus (clean)")
    ap.add_argument("--model", default="google/embeddinggemma-300m", help="ST model name")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if None)")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"], help="GPU half precision if available")
    ap.add_argument("--out-npy", required=True, type=pathlib.Path)
    ap.add_argument("--out-meta", required=True, type=pathlib.Path)
    ap.add_argument("--out-latency-csv", default=pathlib.Path("artifacts/reports/latency_embed.csv"), type=pathlib.Path)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Optional: load .env if present (no hard dependency)
    dotenv_path = pathlib.Path(".env")
    if dotenv_path.exists():
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(dotenv_path)
        except Exception:
            pass

    # If model is gated, use HUGGINGFACE_HUB_TOKEN env var
    if "HUGGINGFACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_HUB_TOKEN"]

    embed_corpus(
        input_jsonl=args.input,
        model_name=args.model,
        batch_size=args.batch_size,
        out_npy=args.out_npy,
        out_meta=args.out_meta,
        out_latency_csv=args.out_latency_csv,
        device=args.device,
        precision=args.precision,
    )
