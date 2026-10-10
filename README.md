# mini-moe

Identify trading cards from real photos by matching them against a catalog of
Scryfall images using an embedding index.

## Layout

```
data/
  images/          catalog images from Scryfall (gitignored)
  test_photos/     hand-collected real photos (gitignored)
    labels.csv     columns: filename, card_name, set
scripts/
  01_build_catalog.py   build the catalog from Scryfall
  02_build_index.py     embed catalog images and build the FAISS index
  03_identify.py        identify cards in test photos
notes/journal.md        daily learning log
notebooks/              scratch Jupyter notebooks
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the scripts in order:

```bash
python scripts/01_build_catalog.py
python scripts/02_build_index.py
python scripts/03_identify.py
```
