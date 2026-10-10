"""
STEP 3 — Give it a phone photo, get the top-5 matches and a rough confidence score.

What it does:
  1. Computes the query image's embedding (exactly the same pipeline used for the catalog — this is
     essential, otherwise the vectors won't be comparable).
  2. Top-K search in FAISS.
  3. Confidence v0 = the "margin" top1_score - top2_score. A small margin means the model is
     confused between two printings → "not sure". In Week 4 we will replace this with proper calibration.

Usage:  python scripts/03_identify.py path/to/photo.jpg
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
step2 = import_module("02_build_index")

import numpy as np
import pandas as pd
import faiss

DATA = Path("data")
TOP_K = 5
MARGIN_THRESHOLD = 0.03   # a guess for now; will tune it with the Week 2 test set


def main(photo):
    df = pd.read_parquet(DATA / "catalog.parquet")
    index = faiss.read_index(str(DATA / "index.faiss"))
    processor, model = step2.load_model()

    q = step2.embed_images([photo], processor, model)     # (1, D)
    scores, ids = index.search(q, TOP_K)                  # both (1, K)
    scores, ids = scores[0], ids[0]

    margin = float(scores[0] - scores[1])
    verdict = "CONFIDENT" if margin > MARGIN_THRESHOLD else "NOT SURE — showing candidates"
    print(f"\n{verdict}  (top1={scores[0]:.3f}, margin={margin:.3f})\n")

    for rank, (s, i) in enumerate(zip(scores, ids), 1):
        r = df.iloc[int(i)]
        price = f"${r.price_usd}" if pd.notna(r.price_usd) else "n/a"
        print(f"{rank}. {s:.3f}  {r['name']}  [{r.set.upper()} #{r.collector_number}, {r.released_at}]  {price}")


if __name__ == "__main__":
    main(sys.argv[1])