"""
STEP 2 — Compute an embedding for every catalog image and build a FAISS index.

What it does:
  1. Loads DINOv2 (Meta's self-supervised Vision Transformer). It has seen 140M images without
     labels, so even without any training it is good at judging "are these two images similar or not".
     It is the image version of a text sentence-embedding model.
  2. Each image → 384-dim vector (CLS token), L2-normalized.
     After normalizing, dot product = cosine similarity.
  3. All vectors go into a FAISS IndexFlatIP (exact search, inner product).
     Backend analogy: FAISS = pgvector, just in-memory and very fast.

Output: data/index.faiss + data/embeddings.npy (the row order of catalog.parquet is the index order).
"""
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DATA = Path("data")
MODEL_NAME = "facebook/dinov2-small"   # runs on CPU; if you have a GPU, try dinov2-base
BATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def embed_images(paths, processor, model):
    """List of images → (N, D) normalized numpy array."""
    out = []
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        # last_hidden_state: (B, tokens, D). Token 0 = CLS = summary vector of the whole image.
        cls = model(**inputs).last_hidden_state[:, 0]
        cls = torch.nn.functional.normalize(cls, dim=-1)
        out.append(cls.cpu().numpy().astype("float32"))
        if (i // BATCH) % 10 == 0:
            print(f"{i}/{len(paths)}")
    return np.concatenate(out)


def main():
    df = pd.read_parquet(DATA / "catalog.parquet")
    processor, model = load_model()
    emb = embed_images(df["image_path"].tolist(), processor, model)
    print("embeddings:", emb.shape)

    index = faiss.IndexFlatIP(emb.shape[1])   # IP = inner product; normalized vectors → cosine
    index.add(emb)
    faiss.write_index(index, str(DATA / "index.faiss"))
    np.save(DATA / "embeddings.npy", emb)
    print("index built with", index.ntotal, "vectors")


if __name__ == "__main__":
    main()