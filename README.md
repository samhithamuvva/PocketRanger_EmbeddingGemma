````markdown
# Pocket Ranger — Evaluating Matryoshka Representation Learning (MRL) in EmbeddingGemma

This project examines **Matryoshka Representation Learning (MRL)** in Google’s **EmbeddingGemma** model. It measures how performance and retrieval consistency vary across embedding slices (128 / 256 / 512 / 768D) using a National Parks corpus and FAISS-based retrieval.

The repository supports:
1. Direct evaluation with pre-generated embeddings and FAISS indices.  
2. Full pipeline execution: corpus collection → embedding generation → index building → retrieval visualization.  
3. A Streamlit app for comparing Matryoshka slice performance side-by-side.

---

**## 1. Quick Start (Direct Use)**

```powershell
cd "Your folder path"
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Refer Model Loading Section to load the LLM or connect to your LLM of choice.
streamlit run app\streamlit_app.py
````

---

## 2. Full Workflow (Build Everything)

### (a) Data Collection

```powershell
cd "Your folder path""
python -m venv .venv && .\.venv\Scripts\Activate.ps1 && pip install -r requirements.txt
python scripts\collect_parks_corpus.py --parks yose seki grca
```

---

### (b) Embedding Generation

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\embed_corpus.py --input data\clean\parks_corpus.jsonl --model google/embeddinggemma-300m --batch-size 16 --out-npy artifacts\embeddings\parks_768_fp16.npy --out-meta artifacts\embeddings\meta.jsonl
```

---

### (c) Build Matryoshka Slices

```powershell
python scripts\build_indices.py --embeddings artifacts\embeddings\parks_768_fp16.npy --meta artifacts\embeddings\meta.jsonl --out-dir artifacts\faiss --dims 128 256 512 768 --save-slices
```

---

### (d) Retrieval Demo (to verify the retrieval)

```powershell
python scripts\query_demo.py --index artifacts\faiss\parks_flatip_256.index --meta artifacts\embeddings\meta.jsonl --k 5 --probe 50
```

---

## 3. LLM Setup (Gemma)

```powershell
pip install llama-cpp-python
huggingface-cli login
huggingface-cli download bartowski/gemma-2-2b-it-GGUF gemma-2-2b-it-Q8_0.gguf --local-dir ./models
```

These models are automatically loaded by the Streamlit interface for query-response testing.

---

## 4. Overlap Evaluation (MRL Slice Comparison)

```powershell
pip install scikit-learn && pip install ujson
python scripts\test_matryoshka_overlap.py
```

---

## 5. Streamlit Interface

Two UI options are available:

```powershell
streamlit run scripts\app.py
```

The app visualizes retrieval variance and overlap across embedding dimensions, with a strict mode enforcing nested slice comparison.

---

## 6. Repository Structure

```
pocket_ranger/                     ]
├─ scripts/                   # Data, embedding, FAISS, and all the scripts
├─ data/                      # Raw and cleaned corpus
├─ artifacts/                 # Generated embeddings and FAISS indices
├─ models/                    # Local GGUF models (Gemma) Create one in your folder
├─ requirements.txt           # Dependencies
├─ .env.example               # Template for keys
├─ .gitignore                 # Ignore environment and model files
└─ README.md
```

---

## 7. Notes

* Activate `.venv` before running any script.
* Ensure valid `NPS_API_KEY` and `HUGGINGFACE_HUB_TOKEN` are set.
* CUDA 12.1 GPU setup is optional but improves embedding generation speed.
* Once embeddings and models are downloaded, the app can run fully offline.

---

**Author:** Samhitha Muvva
**Purpose:** Empirically test Matryoshka Representation Learning (MRL) behavior of EmbeddingGemma across multi-dimension slices using a realistic parks corpus.

```


```
