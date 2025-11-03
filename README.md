# Pocket Ranger: Evaluating Matryoshka Representation Learning (MRL) in EmbeddingGemma

This project investigates **Matryoshka Representation Learning (MRL)** behavior in Google’s **EmbeddingGemma** model. It evaluates how retrieval quality and representation compactness vary across embedding slice dimensions (128 / 256 / 512 / 768D) using a real-world National Parks corpus and FAISS-based retrieval. 
The corpus is built using the U.S. National Park Service (NPS) public API, which provides structured data about park details, alerts, places, activities, campgrounds, and events. For this study, data was collected from Yosemite (YOSE), Sequoia (SEKI), and Grand Canyon (GRCA) national parks.
Each endpoint’s JSON response is parsed, cleaned, and consolidated into a unified document-level corpus stored at data/clean/parks_corpus.jsonl

The repository supports:
1. **Direct use** — Evaluate using pre-generated embeddings and indices.  
2. **Full pipeline execution** — From corpus collection → embedding generation → FAISS indexing → retrieval comparison.  
3. **Streamlit visualization** — Compare Matryoshka slice performance across dimensions interactively.


## 1. Quick Start (Direct Use)

```powershell
cd "C:\Users\samhi\OneDrive - sjsu.edu\Documents\sjsu\Blog\EmbeddingGemma\pocket_ranger"
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
````

Refer to the **Model Loading** section below to connect your LLM (Gemma or any other compatible GGUF model).

Then launch the Streamlit interface:

```powershell
streamlit run app\streamlit_app.py
```

---

## 2. Full Workflow (From Scratch)

### (a) Data Collection

```powershell
cd "C:\Users\samhi\OneDrive - sjsu.edu\Documents\sjsu\Blog\EmbeddingGemma\pocket_ranger"
python -m venv .venv && .\.venv\Scripts\Activate.ps1 && pip install -r requirements.txt
$env:NPS_API_KEY="your_api_key_here"
python scripts\collect_parks_corpus.py --parks yose seki grca
```

This script collects and cleans National Park Service (NPS) data into a consolidated corpus at:

```
data\clean\parks_corpus.jsonl
```

---

### (b) Embedding Generation

Activate your virtual environment and ensure the required library versions for **EmbeddingGemma 3 compatibility**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then generate embeddings:

```powershell
python scripts\embed_corpus.py --input data\clean\parks_corpus.jsonl --model google/embeddinggemma-300m --batch-size 16 --out-npy artifacts\embeddings\parks_768_fp16.npy --out-meta artifacts\embeddings\meta.jsonl
```

This produces:

* `parks_768_fp16.npy` — embeddings matrix (768D)
* `meta.jsonl` — metadata aligned with embedding vectors

---

### (c) Build Matryoshka Slices (FAISS Indices)

```powershell
python scripts\build_indices.py --embeddings artifacts\embeddings\parks_768_fp16.npy --meta artifacts\embeddings\meta.jsonl --out-dir artifacts\faiss --dims 128 256 512 768 --save-slices
```

This creates multiple FAISS indices (`FlatIP`) to enable Matryoshka slice comparisons across embedding sizes.

---

### (d) Retrieval Demo

```powershell
python scripts\query_demo.py --index artifacts\faiss\parks_flatip_256.index --meta artifacts\embeddings\meta.jsonl --k 5 --probe 50
```

The script queries the FAISS index and returns top results to evaluate retrieval stability across slices.

---

## 3. Model Loading (LLM Setup)

Install and configure **Llama.cpp** for local inference:

```powershell
pip install llama-cpp-python
huggingface-cli login
```

Enter your Hugging Face access token (for gated Gemma models):

```
hf token: <your_token_here>
```

Download your preferred model (choose one):

```powershell
huggingface-cli download bartowski/gemma-2-2b-it-GGUF gemma-2-2b-it-Q8_0.gguf --local-dir ./models
```

These models are automatically loaded by the Streamlit interface for generation and response comparison.

---

## 4. Evaluation — Slice Overlap & MRL Consistency

Install dependencies and run the overlap test:

```powershell
python scripts\test_matryoshka_overlap.py
```

This script computes cosine-similarity overlaps across slices (128→768D) to measure representational consistency — a key indicator of Matryoshka behavior.

---

## 5. Launch the Streamlit Interface

Run the interactive interface to visualize and compare results:

```powershell
streamlit run scripts\app.py
```

or for stricter MRL evaluation mode:

```powershell
streamlit run scripts\app_strict_matryoshka.py
```

The app provides:

* Query-based RAG evaluation for multiple embedding sizes
* Strict Matryoshka alignment option
* Slice-wise similarity and overlap visualization

---

## 6. Repository Structure

```
pocket_ranger/
├─ app/                       # Streamlit visualization UI
├─ scripts/                   # Data collection, embedding, FAISS, and evaluation scripts
├─ data/                      # Raw and cleaned National Park corpus
├─ artifacts/                 # Generated embeddings and FAISS indices
├─ models/                    # Local GGUF models (Gemma / TinyLlama)
├─ requirements.txt           # Python dependencies
├─ .env.example               # Template for API keys and tokens
├─ .gitignore                 # Excludes large/local files (.venv, artifacts, etc.)
└─ README.md
```

---

## 7. Environment Variables

All credentials can be set via `.env` or inline commands:

```powershell
$env:NPS_API_KEY="your_api_key_here"
$env:HUGGINGFACE_HUB_TOKEN="your_hf_token_here"
$env:HF_TOKEN=$env:HUGGINGFACE_HUB_TOKEN
```

---

## 8. Notes

* Always activate `.venv` before running any script.
* Ensure correct Hugging Face token access for Gemma models.
* CUDA 12.1 is optional but improves embedding generation speed.
* Once embeddings and indices are generated, the app runs offline.

---

**Author:** Samhitha Muvva

