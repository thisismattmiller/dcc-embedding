# DDC × Embedding Space

What does the Dewey Decimal Classification look like if you trace it as a curve through a modern text-embedding space? Does walking from class 000 → 999 in numeric order mean walking smoothly through topical meaning ordering jump around?

This repo runs that experiment over ~5.6 million catalogued books from the Library of Congress.

**Live viewer:** https://thisismattmiller.github.io/dcc-embedding/

## The Pipeline

The scripts run in order. Each step writes its outputs to `/Volumes/ImNotGlum/ddc_embedding/` (large data lives outside the repo).

| # | Script | What it does |
|---|---|---|
| 1 | [`extract_ddc.py`](extract_ddc.py) | Streams 20.3M MARC records from `bibs.mrc`, pulls the Dewey number from MARC field 082 $a, writes `ddc.tsv` (~5.6M books with a usable DDC). |
| 2 | [`extract_ddc_labels.py`](extract_ddc_labels.py) | Parses `ddc-wikipedia.html` (a saved copy of Wikipedia's "List of Dewey Decimal classes") and writes `ddc_labels.tsv` with the English label for each 3-digit class. |
| 3 | [`phase1_join.py`](phase1_join.py) | Joins the DDC table to the 1024-dim text embeddings from `rag_embeddings/*.jsonl`. Writes parallel numpy arrays: `vectors.f32.npy` (5.6M × 1024), `ddc.f64.npy`, and `ids.txt`. |
| 4 | [`phase2_centroids.py`](phase2_centroids.py) | Groups books by 3-digit DDC class (000–999), averages the unit-vector embeddings per class, re-normalizes. Writes `centroids.f32.npy` (996 × 1024) plus `class_ids.i32.npy` and `counts.i64.npy`. |
| 5 | [`phase3_plot.py`](phase3_plot.py) | The static plots. Computes consecutive-step cosine distances along the DDC ordering, renders `smoothness.png` (the headline curve), `map_pca.png`, `map_umap.png`, and `discontinuities.tsv` (top 30 jumps with labels). |
| 6 | [`phase4_web.py`](phase4_web.py) | The interactive viewer. Re-runs PCA + UMAP and writes a single self-contained `docs/index.html` with both projections as zoomable SVG, count-sized points, collision-aware label tiering, and hover tooltips. |

## Inputs

- `ddc-wikipedia.html` — a saved page of [List of Dewey Decimal classes](https://en.wikipedia.org/wiki/List_of_Dewey_Decimal_classes), used only as the label source.
- `/Volumes/Glum/lc_bibs/bibs.mrc` — the LC bib export.
- `/Volumes/Glum/lc_bibs/rag_embeddings/*_embeddings.jsonl` — pre-computed 1024-dim embeddings, one per book, keyed by MARC 001.

## Outputs

- **[`discontinuities.tsv`](discontinuities.tsv)** — top 30 places where consecutive Dewey classes are far apart in embedding space. The biggest jumps are mostly the 100s-block seams (519→520, 399→400, 799→800) where Dewey's numeric ordering switches topics with no transition.
- **[`smoothness.png`](smoothness.png)** — distance vs. DDC class. Mostly low, with sharp spikes at the seams.
- **[`docs/index.html`](docs/index.html)** — the zoomable PCA / UMAP viewer ([live](https://thisismattmiller.github.io/dcc-embedding/)).


