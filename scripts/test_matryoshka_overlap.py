import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def overlap_score(X_full, ks=(128,256,512), topk=10):
    sims_full = cosine_similarity(X_full)
    full_top = np.argsort(-sims_full, axis=1)[:, 1:topk+1]
    for d in ks:
        sims_d = cosine_similarity(X_full[:, :d])
        top_d = np.argsort(-sims_d, axis=1)[:, 1:topk+1]
        overlap = (np.isin(top_d, full_top)).mean()
        print(f"{d}D vs 768D overlap: {overlap*100:.1f}%")

if __name__ == "__main__":
    path = r"artifacts\embeddings\parks_768_fp16.npy"   # or fp16 if that’s what you have
    X = np.load(path)
    if X.shape[0] > 2000:
        X = X[:2000]
    X = X.astype(np.float32)
    overlap_score(X)
