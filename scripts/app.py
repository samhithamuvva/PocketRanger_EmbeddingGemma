# app_rag_compare_v2.py
# Streamlit app to compare RAG answers across Matryoshka embedding slices (128/256/512/768D),
# with: Gemma chat formatting, stable policy tie-break, strict Matryoshka mode, and guardrails.
#
# Run:
#   streamlit run app_rag_compare_v2.py
#
# Requires:
#   pip install streamlit faiss-cpu sentence-transformers torch llama-cpp-python psutil regex

import os
import json
import time
import pathlib
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import psutil
import faiss  # type: ignore
import torch
from sentence_transformers import SentenceTransformer  # type: ignore
from llama_cpp import Llama  # type: ignore


# ------------------------- Utils -------------------------
def l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def read_jsonl(path: pathlib.Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def safe_excerpt(text: str, limit: int = 700) -> str:
    if not text:
        return ""
    t = " ".join(text.split())
    return t if len(t) <= limit else t[:limit] + "..."


# ------------------------- Heuristics -------------------------
POLICY_KEYWORDS = [
    "reservation", "reservations", "timed entry", "timed-entry", "permit", "permits",
    "lottery", "quota", "backcountry", "fees", "entrance", "pass", "passes", "hours",
    "seasonal", "open year-round", "shuttle", "parking", "road closure", "opening",
    "closing", "tioga road", "hetch hetchy",
]

# Basic scope classifier and park inference for tie-breaks
def infer_park_code(q: str) -> Optional[str]:
    ql = q.lower()
    if "yosemite" in ql or "tioga" in ql or "half dome" in ql or "hetch hetchy" in ql:
        return "yose"
    if "grand canyon" in ql or "south rim" in ql or "north rim" in ql or "phantom ranch" in ql:
        return "grca"
    if "sequoia" in ql or "kings canyon" in ql or "general sherman" in ql or "grant grove" in ql:
        return "seki"
    return None


# Guardrail: is this likely within U.S. National Parks scope?
SCOPE_HINTS = [
    "national park", "nps", "entrance fee", "campground", "camping", "backcountry",
    "permit", "timed entry", "lodging", "trail", "ranger", "yosemite", "grand canyon",
    "sequoia", "kings canyon",
]

def is_in_scope(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in SCOPE_HINTS)


# ------------------------- Context reduction -------------------------
LINE_HINTS = [
    r"reservation", r"timed[- ]?entry", r"permit", r"lottery", r"quota", r"entrance",
    r"fee", r"pass", r"hours", r"open", r"close", r"season", r"vehicle",
    r"valid for seven consecutive days", r"\$\s?\d+", r"\b\d+\.00\b",
    r"commercial use authorization",
]
LINE_RE = re.compile("|".join(LINE_HINTS), re.IGNORECASE)

def reduce_context(text: str, max_lines: int = 10) -> str:
    lines = re.split(r"[\n\r]+|(?<=\.)\s+", text)
    scored: List[Tuple[int, int, str]] = []
    for i, ln in enumerate(lines):
        ln_c = ln.strip()
        if not ln_c:
            continue
        score = 0
        if LINE_RE.search(ln_c):
            score += 2
        if "http" in ln_c or ":" in ln_c:
            score += 1
        if len(ln_c) < 260:
            score += 1
        scored.append((score, i, ln_c))
    if not scored:
        return safe_excerpt(text, 700)
    scored.sort(key=lambda x: (x[0], -len(x[2])), reverse=True)
    chosen = [t[2] for t in scored[:max_lines]]
    return " ".join(chosen)


# ------------------------- Cached loaders -------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str, device: Optional[str], precision: str) -> SentenceTransformer:
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda" and precision.lower() in ("fp16", "half"):
        try:
            model = model.half()  # type: ignore[attr-defined]
        except Exception:
            pass
    return model


@st.cache_resource(show_spinner=False)
def load_llm(model_path: str, n_ctx: int, n_threads: int, n_gpu_layers: int, verbose: bool) -> Llama:
    # Pick a chat template based on filename (works with TinyLlama, Phi, Gemma)
    mp = model_path.lower()
    if "gemma" in mp:
        chat_fmt = "gemma"
    elif "phi" in mp:
        chat_fmt = "phi3"
    else:
        chat_fmt = "llama-2"  # TinyLlama & most LLaMA-style GGUFs

    try:
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            chat_format=chat_fmt,
        )
    except TypeError:
        # Older llama.cpp wheels without chat_format
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )


@st.cache_resource(show_spinner=False)
def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)


@st.cache_resource(show_spinner=False)
def load_meta(meta_path: str) -> List[Dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_resource(show_spinner=False)
def load_corpus(corpus_jsonl_path: str) -> List[Dict]:
    return read_jsonl(pathlib.Path(corpus_jsonl_path))


# ------------------------- Query encoding (Strict Matryoshka) -------------------------
@st.cache_resource(show_spinner=False)
def encode_query_768(_embedder: SentenceTransformer, query: str) -> np.ndarray:
    """Encode query once at 768D and cache result per unique query string."""
    return _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)


def qslice_norm(q768: np.ndarray, target_dim: int) -> np.ndarray:
    q = q768[:, :target_dim].copy()
    return l2n(q)


# ------------------------- Retrieval & Answering -------------------------
def zscore_inplace(cand: List[Dict]) -> None:
    if not cand:
        return
    xs = np.array([c["score"] for c in cand], dtype=np.float32)
    mu, sd = float(xs.mean()), float(xs.std() + 1e-6)
    for c in cand:
        c["base_score_z"] = (c["score"] - mu) / sd


def rerank_policy(query: str, cand: List[Dict]) -> None:
    policy_types = {"alerts", "parks", "campgrounds", "feespasses", "hours", "articles", "events", "thingstodo"}
    for c in cand:
        bonus = 0.0
        if c["content_type"] == "places":
            bonus -= 0.35
        elif c["content_type"] in policy_types:
            bonus += 0.15
        bt = (c.get("body_text", "") or "").lower()
        hits = sum(1 for k in POLICY_KEYWORDS if k in bt)
        bonus += 0.04 * hits
        park_pref = infer_park_code(query)
        if park_pref and c.get("park_code") == park_pref:
            bonus += 0.15
        c["policy_bonus"] = bonus

    base_sorted = sorted(range(len(cand)), key=lambda i: cand[i]["score"], reverse=True)
    base_rank = {i: r for r, i in enumerate(base_sorted)}
    for i, c in enumerate(cand):
        c["_base_rank"] = base_rank[i]

    # Primary: original FAISS rank; Secondary: policy bonus
    cand.sort(key=lambda c: (c["_base_rank"], -c["policy_bonus"]))


def retrieve(
    idx,
    meta_rows: List[Dict],
    corpus_rows: List[Dict],
    embedder: SentenceTransformer,
    query: str,
    dim: int,
    probe: int,
    top_k: int,
    policy_focus: bool,
    q768_cached: Optional[np.ndarray],
    strict: bool,
) -> List[Dict]:
    if strict and q768_cached is not None:
        qvec = qslice_norm(q768_cached, dim)
    else:
        q768_tmp = encode_query_768(embedder, query)
        qvec = qslice_norm(q768_tmp, dim)

    scores, ids = idx.search(qvec, max(probe, top_k))
    cand: List[Dict] = []
    for sc, ii in zip(scores[0], ids[0]):
        if ii < 0:
            continue
        m = meta_rows[int(ii)]
        body_text = corpus_rows[int(ii)].get("body_text", "")
        cand.append(
            {
                "score": float(sc),
                "id": int(ii),
                "title": m.get("title", ""),
                "url": m.get("url", ""),
                "park_code": m.get("park_code", ""),
                "content_type": (m.get("content_type", "") or "").lower(),
                "body_text": body_text,
            }
        )

    zscore_inplace(cand)
    if policy_focus and not strict:
        rerank_policy(query, cand)
    else:
        cand.sort(key=lambda x: x["score"], reverse=True)
    return cand[:top_k]


def build_prompt_chat(query: str, contexts: List[Dict], max_chars: int = 3200) -> List[Dict]:
    # Guardrail: if contexts are empty, force refusal downstream
    ctx_lines: List[str] = []
    used = 0
    for i, c in enumerate(contexts, start=1):
        reduced = reduce_context(c["body_text"])
        piece = (
            f"[{i}] {c['title']} | {c['content_type']} | {c['park_code']} | {c['url']}\n"
            f"{reduced}\n"
        )
        if used + len(piece) > max_chars:
            break
        ctx_lines.append(piece)
        used += len(piece)

    system = (
        "You are a U.S. National Parks information assistant. "
        "Scope guardrail: If the user's question is not about U.S. National Parks policies, fees, passes, camping, permits, "
        "hours, closures, or official park information, reply exactly with: 'Out of scope: This assistant only answers about U.S. National Parks.' "
        "Grounding rule: Answer ONLY using the provided context. If the context does not contain the answer, reply exactly with: 'Not in context.' "
        "Always include bracket citations like [1], [2] matching the context items you used."
    )

    user = (
        (("Context:\n" + "\n".join(ctx_lines)) if ctx_lines else "Context:\n")
        + (
            f"\nQuestion: {query}\n"
            "Answer concisely in 3–6 sentences. Use precise policy language. Include bracket citations."
        )
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def ask_llm_chat(llm: Llama, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    try:
        out = llm.create_chat_completion(
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            stop=["</s>"],
        )
        txt = (out.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
        return (txt or "").strip()
    except Exception:
        joined = ""
        for m in messages:
            role = m["role"].upper()
            joined += f"{role}:\n{m['content']}\n\n"
        out = llm(
            prompt=joined,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            stop=["</s>", "USER:", "SYSTEM:"],
        )
        return (out.get("choices", [{}])[0].get("text") or "").strip()


# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Pocket Ranger – Matryoshka RAG Comparison", layout="wide")
st.title("Pocket Ranger : Matryoshka RAG Comparison")
st.caption("Strict Matryoshka (single-encode slicing), stable policy tie-break, and scope guardrails.")

# (Optional) Keep fonts simple; using defaults avoids odd glyphs/wrapping.

with st.sidebar:
    st.header("Paths & Models")
    default_faiss_dir = "artifacts/faiss"
    default_meta = "artifacts/embeddings/meta.jsonl"
    default_corpus = "data/clean/parks_corpus.jsonl"
    default_gguf = "scripts/models/gemma-2-2b-it-Q8_0.gguf"
    default_embedder = "google/embeddinggemma-300m"

    faiss_dir = st.text_input("FAISS directory", value=default_faiss_dir)
    meta_path = st.text_input("Meta JSONL path", value=default_meta)
    corpus_path = st.text_input("Clean corpus JSONL path", value=default_corpus)

    st.subheader("Embedder (queries)")
    embedder_name = st.text_input("Sentence-Transformers model", value=default_embedder)
    embedder_device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
    embedder_precision = st.selectbox("Precision", options=["fp16", "fp32"], index=0)

    st.subheader("LLM (answers)")
    gguf_path = st.text_input("GGUF model path", value=default_gguf)
    n_ctx = st.number_input("Context window", min_value=1024, max_value=8192, value=2048, step=256)
    n_threads = st.number_input("n_threads", min_value=1, max_value=os.cpu_count() or 32, value=min(8, os.cpu_count() or 8))
    n_gpu_layers = st.number_input("n_gpu_layers (0=CPU)", min_value=0, max_value=200, value=0)
    llm_verbose = st.checkbox("Verbose logs", value=False)

    st.subheader("Retrieval")
    dims = st.multiselect("Compare dimensions", options=[128, 256, 512, 768], default=[128, 256, 512, 768])
    top_k = st.slider("Top-K per slice", min_value=1, max_value=10, value=3)
    probe = st.slider("Initial probe", min_value=max(3, top_k), max_value=120, value=60)

    # Comparison controls
    strict_mode = st.checkbox("Strict Matryoshka (same query vec, no re-rank)", value=True)
    policy_mode = st.checkbox("Policy-focused re-rank (tie-breaker)", value=False)

    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_tokens = st.slider("Max tokens", 128, 1024, 280, 16)

    st.write(f"RAM available: {round(psutil.virtual_memory().available / (1024 ** 3), 2)} GB")

user_q = st.text_input("Ask a question", value="Do I need a reservation to enter Yosemite in July?")
run_btn = st.button("Compare answers", type="primary")


def get_index_path(dim: int) -> str:
    return str(pathlib.Path(faiss_dir) / f"parks_flatip_{dim}.index")


if run_btn:
    # Guardrail 1: quick pre-filter
    if not is_in_scope(user_q):
        st.error("Out of scope: This assistant only answers about U.S. National Parks.")
        st.stop()

    if not dims:
        st.warning("Select at least one dimension.")
        st.stop()

    # Validate files
    missing: List[str] = []
    for dim in dims:
        if not pathlib.Path(get_index_path(dim)).exists():
            missing.append(get_index_path(dim))
    for p in [meta_path, corpus_path, gguf_path]:
        if not pathlib.Path(p).exists():
            missing.append(p)
    if missing:
        st.error("Missing:\n" + "\n".join(missing))
        st.stop()

    with st.spinner("Loading models & data..."):
        embedder = load_embedder(embedder_name, embedder_device, embedder_precision)
        llm = load_llm(gguf_path, int(n_ctx), int(n_threads), int(n_gpu_layers), bool(llm_verbose))
        meta_rows = load_meta(meta_path)
        corpus_rows = load_corpus(corpus_path)

    if len(meta_rows) != len(corpus_rows):
        st.warning(f"Meta rows ({len(meta_rows)}) != Corpus rows ({len(corpus_rows)}). IDs might misalign.")

    # Compute allowed park codes from meta for guardrail 2
    allowed_park_codes = {(m.get("park_code") or "").lower() for m in meta_rows if m.get("park_code")}

    # Single encode for entire comparison
    q768 = encode_query_768(embedder, user_q)

    st.subheader("Results")
    cols = st.columns(len(dims))
    timings: Dict[int, Dict[str, float]] = {}

    for col, dim in zip(cols, dims):
        with col:
            st.markdown(f"### {dim}D")
            t0 = time.time()
            idx = load_faiss_index(get_index_path(dim))
            t1 = time.time()

            hits = retrieve(
                idx,
                meta_rows,
                corpus_rows,
                embedder,
                user_q,
                dim,
                int(probe),
                int(top_k),
                policy_mode,
                q768_cached=q768,
                strict=bool(strict_mode),
            )

            # Guardrail 2: if retrieved contexts are empty or have no recognized park code, refuse
            if not hits or not any((h.get("park_code") or "").lower() in allowed_park_codes for h in hits):
                st.write("Out of scope: This assistant only answers about U.S. National Parks.")
                timings[dim] = {"load_ms": (t1 - t0) * 1000, "retrieval_ms": 0.0, "gen_ms": 0.0}
                continue

            t2 = time.time()
            messages = build_prompt_chat(user_q, hits)
            answer = ask_llm_chat(llm, messages, float(temperature), int(max_tokens))
            t3 = time.time()

            st.markdown("**Answer**")
            st.write(answer if answer else "(no output)")

            with st.expander("Show retrieved context"):
                st.caption("Raw FAISS IDs (helps compare overlap):")
                st.code([h["id"] for h in hits], language="python")
                for i, h in enumerate(hits, start=1):
                    st.markdown(f"**[{i}] {h['title']}**")
                    # ASCII-only separators
                    st.caption(f"{h['content_type']} | {h['park_code']} | {h['url']}")
                    st.write(safe_excerpt(reduce_context(h["body_text"], max_lines=10), 900))
                    st.divider()

            timings[dim] = {
                "load_ms": (t1 - t0) * 1000,
                "retrieval_ms": (t2 - t1) * 1000,
                "gen_ms": (t3 - t2) * 1000,
            }

            # Structured timing display (no fancy glyphs, no wrapping issues)
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Load (ms)", f"{timings[dim]['load_ms']:.0f}")
            with mcol2:
                st.metric("Retrieval (ms)", f"{timings[dim]['retrieval_ms']:.0f}")
            with mcol3:
                st.metric("Generation (ms)", f"{timings[dim]['gen_ms']:.0f}")

            st.divider()

    # Structured comparison summary as a table
    st.subheader("Comparison Summary")
    summary_rows = []
    for dim in dims:
        t = timings.get(dim, {"load_ms": 0.0, "retrieval_ms": 0.0, "gen_ms": 0.0})
        summary_rows.append(
            {
                "Dim (D)": dim,
                "Load (ms)": f"{t['load_ms']:.0f}",
                "Retrieval (ms)": f"{t['retrieval_ms']:.0f}",
                "Generation (ms)": f"{t['gen_ms']:.0f}",
            }
        )
    st.table(summary_rows)
