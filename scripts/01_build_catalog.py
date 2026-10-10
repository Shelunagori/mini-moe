"""
STEP 1 — Build the catalog (from Scryfall).

What it does:
  1. Downloads Scryfall's "default_cards" bulk JSON (one record per printing).
  2. Keeps only cards that have >= MIN_PRINTINGS distinct printings
     (same card name, different set / frame / art). This is the "fine-grained" problem:
     same name, but different edition → different price.
  3. Downloads the image for each printing and saves it as data/images/<scryfall_id>.jpg.
  4. Saves metadata (name, set, collector_number, price...) to data/catalog.parquet.

Backend analogy: this is your ETL job. Image = row, scryfall_id = primary key.
"""
import json, os, time, random
from pathlib import Path
import gzip
import pandas as pd
import requests

OUT = Path("data")
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

MAX_NAMES = 300        # how many distinct card names to keep (keep it small for speed)
MIN_PRINTINGS = 3      # minimum number of printings a name must have
SEED = 42

# Scryfall API etiquette: User-Agent + Accept are required, plus a ~100ms gap between requests.
HEADERS = {"User-Agent": "MiniMoe/0.1 (learning project)", "Accept": "application/json;q=0.9,*/*;q=0.8"}


def get_bulk_cards():
    """Bulk data is now gzipped JSONL: one card per line. Stream and parse it, caching it once."""
    cache = OUT / "default_cards.jsonl.gz"
    if not cache.exists():
        meta = requests.get("https://api.scryfall.com/bulk-data", headers=HEADERS, timeout=60).json()
        url = next(d["jsonl_download_uri"] for d in meta["data"] if d["type"] == "default_cards")
        print("Downloading bulk JSONL...", url)
        with requests.get(url, headers=HEADERS, timeout=600, stream=True) as resp:
            resp.raise_for_status()
            with open(cache, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    cards = []
    with gzip.open(cache, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cards.append(json.loads(line))
    return cards


def image_url(card):
    """Single-faced cards have image_uris at the top level; double-faced cards have it inside card_faces[0]."""
    if "image_uris" in card:
        return card["image_uris"].get("normal")
    faces = card.get("card_faces") or []
    if faces and "image_uris" in faces[0]:
        return faces[0]["image_uris"].get("normal")
    return None


def main():
    cards = get_bulk_cards()
    print("total printings in bulk:", len(cards))

    rows = []
    for c in cards:
        # Only English, normal paper cards; drop digital-only / tokens / art-series (they are noise).
        if c.get("lang") != "en" or c.get("digital") or c.get("layout") in ("token", "art_series", "emblem"):
            continue
        url = image_url(c)
        if not url:
            continue
        rows.append({
            "scryfall_id": c["id"],
            "name": c["name"],
            "set": c["set"],
            "set_name": c["set_name"],
            "collector_number": c["collector_number"],
            "released_at": c.get("released_at"),
            "frame": c.get("frame"),
            "price_usd": (c.get("prices") or {}).get("usd"),
            "image_url": url,
        })
    df = pd.DataFrame(rows)

    # Fine-grained subset: names that have >= MIN_PRINTINGS printings.
    counts = df.groupby("name").size()
    eligible = counts[counts >= MIN_PRINTINGS].index.tolist()
    random.seed(SEED)
    chosen = set(random.sample(eligible, min(MAX_NAMES, len(eligible))))
    df = df[df["name"].isin(chosen)].reset_index(drop=True)
    print(f"selected {df['name'].nunique()} names, {len(df)} printings")

    # Download images (sequential + sleep: respect Scryfall's rate limit).
    for i, r in df.iterrows():
        path = IMG_DIR / f"{r.scryfall_id}.jpg"
        if path.exists():
            continue
        try:
            resp = requests.get(r.image_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        except Exception as e:
            print("failed", r.scryfall_id, e)
        time.sleep(0.1)
        if i % 100 == 0:
            print(f"{i}/{len(df)} images")

    df["image_path"] = df["scryfall_id"].map(lambda s: str(IMG_DIR / f"{s}.jpg"))
    df = df[df["image_path"].map(os.path.exists)]
    df.to_parquet(OUT / "catalog.parquet", index=False)
    print("saved", OUT / "catalog.parquet", len(df))


if __name__ == "__main__":
    main()