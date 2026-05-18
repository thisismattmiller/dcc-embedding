#!/usr/bin/env python3.12
"""
Phase 2: Aggregate per-book embeddings into per-3-digit-DDC-class centroids.

Reads (from Phase 1):
  - /Volumes/ImNotGlum/ddc_embedding/vectors.f32.npy   (N, 1024) float32, L2-normalized
  - /Volumes/ImNotGlum/ddc_embedding/ddc.f64.npy       (N,)      float64

Drops rows with DDC outside [0, 999] (~0.023% of the input is garbage like
multi-million values from malformed MARC 082 fields). For each integer class
0..999, averages the unit vectors of its members and re-normalizes the result.
The re-normalized mean of unit vectors is the standard "spherical centroid"
— it lives on the unit sphere so cosine distance between centroids is just
1 - dot.

Writes (to /Volumes/ImNotGlum/ddc_embedding/):
  - centroids.f32.npy   (K, 1024) float32, L2-normalized, one row per class
  - class_ids.i32.npy   (K,)      int32   class numbers (0..999) in row order
  - counts.i64.npy      (K,)      int64   book count per class

K is the number of distinct classes present in the valid range (expected ~996).
Row order in all three files is ascending by class id, so they can be indexed
together as parallel arrays.
"""

import os
import sys
import time

import numpy as np

IN_DIR = "/Volumes/ImNotGlum/ddc_embedding"
VECTORS_PATH = os.path.join(IN_DIR, "vectors.f32.npy")
DDC_PATH = os.path.join(IN_DIR, "ddc.f64.npy")

CENTROIDS_PATH = os.path.join(IN_DIR, "centroids.f32.npy")
CLASS_IDS_PATH = os.path.join(IN_DIR, "class_ids.i32.npy")
COUNTS_PATH = os.path.join(IN_DIR, "counts.i64.npy")

DIM = 1024
CHUNK = 200_000  # rows per streaming pass


def fmt_elapsed(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


def main() -> int:
    t0 = time.time()

    print(f"Loading DDC values from {DDC_PATH}", flush=True)
    ddcs = np.load(DDC_PATH, mmap_mode="r")
    n = len(ddcs)
    print(f"  {n:,} rows", flush=True)

    print("Computing class labels and filter mask...", flush=True)
    # Read into RAM — 45 MB, trivial.
    ddcs_arr = np.asarray(ddcs, dtype=np.float64)
    classes_all = np.floor(ddcs_arr).astype(np.int64)
    valid_mask = (classes_all >= 0) & (classes_all <= 999)
    n_valid = int(valid_mask.sum())
    n_dropped = n - n_valid
    print(
        f"  valid: {n_valid:,}  dropped (DDC outside [0,999]): {n_dropped:,}  "
        f"({n_dropped/n*100:.4f}%)",
        flush=True,
    )

    # Allocate accumulators: one sum row per possible class.
    sums = np.zeros((1000, DIM), dtype=np.float64)
    counts = np.zeros(1000, dtype=np.int64)

    print(f"\nStreaming vectors in chunks of {CHUNK:,}...", flush=True)
    vectors = np.load(VECTORS_PATH, mmap_mode="r")
    assert vectors.shape == (n, DIM), f"shape mismatch: {vectors.shape}"

    t_last = time.time()
    PROGRESS_INTERVAL = 5.0

    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        chunk_mask = valid_mask[start:end]
        if not chunk_mask.any():
            continue
        chunk_classes = classes_all[start:end][chunk_mask]
        # Load and cast to float64 for accumulation precision.
        chunk_vecs = np.asarray(vectors[start:end], dtype=np.float64)[chunk_mask]
        # Accumulate into sums grouped by class.
        np.add.at(sums, chunk_classes, chunk_vecs)
        np.add.at(counts, chunk_classes, 1)

        now = time.time()
        if now - t_last >= PROGRESS_INTERVAL or end == n:
            elapsed = now - t0
            pct = end / n * 100
            rate = end / elapsed if elapsed > 0 else 0
            print(
                f"  [{fmt_elapsed(elapsed)}] {end:>10,} / {n:,}  ({pct:.1f}%)  "
                f"rate: {rate:,.0f}/s",
                flush=True,
            )
            t_last = now

    # Keep only classes with at least one member.
    present = counts > 0
    class_ids = np.where(present)[0].astype(np.int32)
    present_counts = counts[present]
    present_sums = sums[present]

    print(f"\nDistinct classes present: {len(class_ids)} / 1000", flush=True)

    # Spherical centroid: mean of unit vectors, then re-normalize.
    means = present_sums / present_counts[:, None]
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    # Guard against zero norm (shouldn't happen with normalized inputs unless
    # cancellation, but be safe — replace 0 with 1 so we don't divide by zero).
    norms[norms == 0] = 1.0
    centroids = (means / norms).astype(np.float32)

    print("\nWriting outputs...", flush=True)
    np.save(CENTROIDS_PATH, centroids)
    np.save(CLASS_IDS_PATH, class_ids)
    np.save(COUNTS_PATH, present_counts)

    print(f"  {CENTROIDS_PATH}   ({centroids.shape[0]} x {DIM} float32)")
    print(f"  {CLASS_IDS_PATH}   ({class_ids.shape[0]} int32)")
    print(f"  {COUNTS_PATH}      ({present_counts.shape[0]} int64)")

    # Quick sanity report.
    print("\nClass count distribution:")
    print(f"  min: {int(present_counts.min()):,}")
    print(f"  median: {int(np.median(present_counts)):,}")
    print(f"  max: {int(present_counts.max()):,}")
    print(f"  total books in centroids: {int(present_counts.sum()):,}")

    # Smoke-check: distance between two adjacent class centroids.
    if len(class_ids) >= 2:
        # Find class 597 and 598 if present (fishes/birds).
        idx = {int(c): i for i, c in enumerate(class_ids)}
        if 597 in idx and 598 in idx:
            d = 1.0 - float(centroids[idx[597]] @ centroids[idx[598]])
            print(f"\n  cosine distance 597->598 (fishes->birds): {d:.4f}")
        if 599 in idx and 600 in idx:
            d = 1.0 - float(centroids[idx[599]] @ centroids[idx[600]])
            print(f"  cosine distance 599->600 (mammals->technology): {d:.4f}")
        if 812 in idx and 813 in idx:
            d = 1.0 - float(centroids[idx[812]] @ centroids[idx[813]])
            print(f"  cosine distance 812->813 (Am. drama->Am. fiction): {d:.4f}")

    print(f"\nDone in {fmt_elapsed(time.time() - t0)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
