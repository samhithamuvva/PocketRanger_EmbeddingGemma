#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build FAISS indices for Matryoshka slices (128/256/512/768) from 768-D embeddings.

Inputs:
  artifacts/embeddings/parks_768_fp16.npy   # [N, 768], float16 or float32, already L2-normalized
  artifacts/embeddings/meta.jsonl           # aligned metadata for reference (not required by FAISS)

Outputs:
  artifacts/faiss/parks_flatip_128.index
  artifacts/faiss/parks_flatip_256.index
  artifacts/faiss/parks_flatip_512.index
  artifacts/faiss/parks_flatip_768.index
  artifacts/reports/index_sizes.csv         # rows: dim, num_vecs, bytes, MB

Usage:
  python scripts/build_indices.py \
    --embeddings artifacts/embeddings/parks_768_fp16.npy \
    --meta artifacts/embeddings/meta.jsonl \
    --out-dir artifacts/faiss \
    --dims 128 256 512 768 \
    --save-slices  # optional; also writes .npy for each slice
"""

import argparse
import json
import pathlib
import os
import numpy as np
import faiss
from typing import List


def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # safety: recompute norms post-slice
    v = mat.astype(np.float32, copy=False)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    v = v / norms
    return v.astype(np.float32, copy=False)


def sizeof(path: pathlib.Path) -> int:
    return path.stat().st_size if path.exists() else 0


def build_index_for_dim(full_vecs: np.ndarray, D: int, out_path: pathlib.Path) -> int:
    # Slice first D dims (Matryoshka protocol) and renormalize
    sliced = full_vecs[:, :D]
    sliced = l2_normalize(sliced)

    # Build FlatIP (cosine when vectors are L2-normalized)
    index = faiss.IndexFlatIP(D)
    index.add(sliced.astype(np.float32))
    faiss.write_index(index, str(out_path))
    return sliced.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, type=pathlib.Path, help="Path to base 768-D .npy file")
    ap.add_argument("--meta", required=False, type=pathlib.Path, help="Aligned metadata JSONL (optional)")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path, help="Directory to write FAISS indices")
    ap.add_argument("--dims", nargs="+", type=int, default=[128, 256, 512, 768], help="Slice dimensions to build")
    ap.add_argument("--save-slices", action="store_true", help="Also save sliced .npy arrays for inspection")
    ap.add_argument("--reports-dir", type=pathlib.Path, default=pathlib.Path("artifacts/reports"))
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.reports_dir)

    # Load base embeddings (N, 768)
    vecs = np.load(args.embeddings)
    if vecs.ndim != 2 or vecs.shape[1] < 768:
        raise SystemExit(f"Expected shape (N, >=768), got {vecs.shape}")

    # Sort and uniquify dims, clamp to available width
    dims: List[int] = sorted(set(int(d) for d in args.dims if 1 <= int(d) <= vecs.shape[1]))
    if not dims:
        raise SystemExit("No valid dims to build indices for.")

    # Build each index
    size_rows = []
    for D in dims:
        out_idx = args.out_dir / f"parks_flatip_{D}.index"
        print(f"[Build] D={D} → {out_idx}")
        n = build_index_for_dim(vecs, D, out_idx)

        bytes_on_disk = sizeof(out_idx)
        size_rows.append({
            "dim": D,
            "num_vecs": n,
            "bytes": bytes_on_disk,
            "MB": round(bytes_on_disk / (1024**2), 3)
        })

        if args.save_slices:
            out_npy = args.out_dir / f"parks_slice_{D}.npy"
            # Re-slice & normalize once more for saving
            sliced = l2_normalize(vecs[:, :D])
            np.save(out_npy, sliced.astype(np.float32))
            print(f"  [Saved slice] {out_npy} ({sliced.shape})")

    # Write size report
    sizes_csv = args.reports_dir / "index_sizes.csv"
    new_file = not sizes_csv.exists()
    import csv
    with sizes_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dim", "num_vecs", "bytes", "MB"])
        if new_file:
            w.writeheader()
        for row in size_rows:
            w.writerow(row)
    print(f"[Report] {sizes_csv} (appended {len(size_rows)} rows)")


if __name__ == "__main__":
    main()
