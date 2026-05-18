#!/usr/bin/env python3.12
"""
Phase 3: Plot the curve of Dewey Decimal classes through embedding space.

Reads (from Phase 2):
  - /Volumes/ImNotGlum/ddc_embedding/centroids.f32.npy
  - /Volumes/ImNotGlum/ddc_embedding/class_ids.i32.npy
  - /Volumes/ImNotGlum/ddc_embedding/counts.i64.npy
  - /Volumes/ImNotGlum/ddc_embedding/ddc_labels.tsv

Produces (in the current working directory):
  - smoothness.png       : consecutive-step cosine distance vs DDC class
  - map_pca.png          : 2D PCA projection of centroids, line in DDC order
  - map_umap.png         : same, but UMAP projection
  - discontinuities.tsv  : top-N largest jumps with class labels

Centroid filter: only classes with >= MIN_COUNT books are kept, so noisy
sparse-class centroids don't inflate the jumps we report.
"""

import os
import sys
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from sklearn.decomposition import PCA

IN_DIR = "/Volumes/ImNotGlum/ddc_embedding"
CENTROIDS_PATH = os.path.join(IN_DIR, "centroids.f32.npy")
CLASS_IDS_PATH = os.path.join(IN_DIR, "class_ids.i32.npy")
COUNTS_PATH = os.path.join(IN_DIR, "counts.i64.npy")
LABELS_PATH = os.path.join(IN_DIR, "ddc_labels.tsv")

MIN_COUNT = 20
TOP_JUMPS = 30


def fmt_elapsed(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m:d}m{sec:02d}s"


def load_labels(path: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            try:
                out[int(parts[0])] = parts[1]
            except ValueError:
                continue
    return out


def label_for(class_id: int, labels: dict[int, str]) -> str:
    return labels.get(class_id, f"({class_id:03d} no label)")


def plot_smoothness(class_ids, distances, labels, out_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    midpoints = (class_ids[:-1] + class_ids[1:]) / 2
    ax.plot(midpoints, distances, linewidth=0.8, color="#222")
    ax.set_xlabel("DDC class (midpoint of consecutive pair)")
    ax.set_ylabel("cosine distance between consecutive centroids")
    ax.set_title(
        f"Smoothness of the DDC curve through embedding space\n"
        f"({len(class_ids)} classes, min {MIN_COUNT} books each)"
    )
    # Mark major Dewey boundaries (every 100).
    for x in range(100, 1000, 100):
        ax.axvline(x, color="#bbb", linestyle=":", linewidth=0.5, zorder=0)
    # Label the top jumps.
    top_idx = np.argsort(distances)[::-1][:8]
    for i in top_idx:
        a, b = int(class_ids[i]), int(class_ids[i + 1])
        ax.annotate(
            f"{a:03d}→{b:03d}",
            xy=(midpoints[i], distances[i]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#a00",
        )
    ax.set_xlim(0, 1000)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_map(coords_2d, class_ids, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    norm = Normalize(vmin=0, vmax=999)
    cmap = plt.get_cmap("turbo")

    # Line segments between consecutive class centroids, colored by DDC midpoint.
    segments = np.stack([coords_2d[:-1], coords_2d[1:]], axis=1)
    midpoints = (class_ids[:-1] + class_ids[1:]) / 2
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        array=midpoints,
        linewidth=0.8,
        alpha=0.75,
    )
    ax.add_collection(lc)

    ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=class_ids,
        cmap=cmap,
        norm=norm,
        s=8,
        zorder=3,
        edgecolors="none",
    )

    # Label every 100s as anchor points.
    for k, cid in enumerate(class_ids):
        if cid % 100 == 0:
            ax.annotate(
                f"{int(cid):03d}",
                xy=(coords_2d[k, 0], coords_2d[k, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                color="#000",
            )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02
    )
    cbar.set_label("DDC class")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def write_discontinuities(class_ids, distances, labels, out_path, top_n):
    order = np.argsort(distances)[::-1][:top_n]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("rank\tfrom_class\tto_class\tdistance\tfrom_label\tto_label\n")
        for rank, i in enumerate(order, start=1):
            a, b = int(class_ids[i]), int(class_ids[i + 1])
            f.write(
                f"{rank}\t{a:03d}\t{b:03d}\t{float(distances[i]):.4f}\t"
                f"{label_for(a, labels)}\t{label_for(b, labels)}\n"
            )
    print(f"  wrote {out_path}", flush=True)


def main() -> int:
    t0 = time.time()

    print("Loading centroids...", flush=True)
    centroids = np.load(CENTROIDS_PATH)
    class_ids = np.load(CLASS_IDS_PATH)
    counts = np.load(COUNTS_PATH)
    labels = load_labels(LABELS_PATH)
    print(
        f"  centroids: {centroids.shape}  classes: {len(class_ids)}  "
        f"labels: {len(labels)}",
        flush=True,
    )

    # Filter by min book count.
    keep = counts >= MIN_COUNT
    centroids = centroids[keep]
    class_ids = class_ids[keep]
    counts = counts[keep]
    print(
        f"  after >= {MIN_COUNT}-book filter: {len(class_ids)} classes "
        f"(dropped {int((~keep).sum())})",
        flush=True,
    )

    # Sort by class id (should already be, but make sure).
    order = np.argsort(class_ids)
    centroids = centroids[order]
    class_ids = class_ids[order]
    counts = counts[order]

    # ── Smoothness curve ────────────────────────────────────────────
    print("\nComputing consecutive-step cosine distances...", flush=True)
    # centroids are unit vectors -> cosine dist = 1 - dot
    dots = np.einsum("ij,ij->i", centroids[:-1], centroids[1:])
    dots = np.clip(dots, -1.0, 1.0)
    distances = 1.0 - dots
    print(
        f"  mean: {distances.mean():.4f}  median: {np.median(distances):.4f}  "
        f"p95: {np.percentile(distances, 95):.4f}  max: {distances.max():.4f}",
        flush=True,
    )

    plot_smoothness(class_ids, distances, labels, "smoothness.png")
    write_discontinuities(class_ids, distances, labels, "discontinuities.tsv", TOP_JUMPS)

    # ── PCA map ────────────────────────────────────────────────────
    print("\nFitting PCA (2D)...", flush=True)
    pca = PCA(n_components=2, random_state=0)
    pca_coords = pca.fit_transform(centroids)
    print(
        f"  explained variance: "
        f"{pca.explained_variance_ratio_[0]*100:.1f}% + "
        f"{pca.explained_variance_ratio_[1]*100:.1f}% = "
        f"{pca.explained_variance_ratio_.sum()*100:.1f}%",
        flush=True,
    )
    plot_map(
        pca_coords,
        class_ids,
        f"DDC curve through embedding space (PCA, {len(class_ids)} classes)",
        "map_pca.png",
    )

    # ── UMAP map ───────────────────────────────────────────────────
    print("\nFitting UMAP (2D)...", flush=True)
    import umap  # imported here so PCA still works if umap is missing

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=0,
    )
    umap_coords = reducer.fit_transform(centroids)
    plot_map(
        umap_coords,
        class_ids,
        f"DDC curve through embedding space (UMAP, {len(class_ids)} classes)",
        "map_umap.png",
    )

    print(f"\nDone in {fmt_elapsed(time.time() - t0)}.")
    print("\nTop 10 discontinuities:")
    order = np.argsort(distances)[::-1][:10]
    for rank, i in enumerate(order, start=1):
        a, b = int(class_ids[i]), int(class_ids[i + 1])
        print(
            f"  {rank:2d}. {a:03d}→{b:03d}  d={distances[i]:.4f}   "
            f"{label_for(a, labels)}  →  {label_for(b, labels)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
