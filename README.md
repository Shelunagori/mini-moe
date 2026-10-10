# Mini-Moe — fine-grained card identification (v0 retrieval)

Goal: given a phone photo of a Magic: The Gathering card, identify the exact *printing*
(name + set + collector number), not just the card name — because different printings
of the same card have very different resale prices.

## Week 1 pipeline (no training)
1. `01_build_catalog.py` — Scryfall bulk data → subset of cards with ≥3 printings → download images + metadata.
2. `02_build_index.py` — DINOv2 embeddings for every catalog image → FAISS index.
3. `03_identify.py photo.jpg` — embed the photo, nearest-neighbour search, top-5 + margin-based confidence.

## Run
Run from the repo root (the scripts read and write `data/` relative to the current directory).
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_build_catalog.py      # ~10-15 min (downloads ~1-2k images)
python scripts/02_build_index.py        # CPU: a few minutes for 1-2k images
python scripts/03_identify.py my_photo.jpg
```

## Honest notes
- Test photos are of *printed* card images (no physical cards available), shot under varied lighting/angles.
- v0 confidence is a crude top1-top2 margin; proper calibration comes in Week 4.
