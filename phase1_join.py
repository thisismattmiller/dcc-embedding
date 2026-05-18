#!/usr/bin/env python3.12
"""
Phase 1: Join DDC table to embedding vectors.

Reads:
  - /Volumes/ImNotGlum/ddc_embedding/ddc.tsv  (001 -> ddc, from extract_ddc.py)
  - /Volumes/ImNotGlum/lc_bibs/rag_embeddings/*_embeddings.jsonl  (27 files)

Writes (all in /Volumes/ImNotGlum/ddc_embedding/):
  - vectors.f32.npy   shape (N, 1024)  float32
  - ddc.f64.npy       shape (N,)       float64
  - ids.txt           N lines of lc_001 in row order

Streams JSONL and appends to memory-mapped numpy arrays so peak RAM stays low.
Sized to a hard upper bound from the DDC table (one row per matched 001), then
truncated to the actual hit count.
"""

import glob
import json
import os
import sys
import time

import numpy as np

DDC_TSV = "/Volumes/ImNotGlum/ddc_embedding/ddc.tsv"
EMB_GLOB = "/Volumes/ImNotGlum/lc_bibs/rag_embeddings/*_embeddings.jsonl"
OUT_DIR = "/Volumes/ImNotGlum/ddc_embedding"

VECTORS_PATH = os.path.join(OUT_DIR, "vectors.f32.npy")
DDC_PATH = os.path.join(OUT_DIR, "ddc.f64.npy")
IDS_PATH = os.path.join(OUT_DIR, "ids.txt")

DIM = 1024
PROGRESS_INTERVAL = 5.0


def fmt_elapsed(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


def load_ddc_table(path: str) -> dict[str, float]:
    print(f"Loading DDC table from {path}", flush=True)
    t = time.time()
    table: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("001\t"):
            raise RuntimeError(f"unexpected header: {header!r}")
        for line in f:
            tab = line.find("\t")
            if tab <= 0:
                continue
            lc_001 = line[:tab]
            ddc_str = line[tab + 1:].rstrip("\n")
            try:
                table[lc_001] = float(ddc_str)
            except ValueError:
                continue
    print(f"  {len(table):,} entries in {fmt_elapsed(time.time() - t)}", flush=True)
    return table


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    ddc_table = load_ddc_table(DDC_TSV)
    max_rows = len(ddc_table)

    files = sorted(glob.glob(EMB_GLOB))
    if not files:
        print(f"No embedding files matched: {EMB_GLOB}", file=sys.stderr)
        return 1
    print(f"\nFound {len(files)} embedding files.", flush=True)

    # Pre-allocate memmaps sized to the upper bound (one match per DDC entry).
    print(f"Allocating output arrays for up to {max_rows:,} rows...", flush=True)
    vectors = np.lib.format.open_memmap(
        VECTORS_PATH, mode="w+", dtype=np.float32, shape=(max_rows, DIM)
    )
    ddcs = np.lib.format.open_memmap(
        DDC_PATH, mode="w+", dtype=np.float64, shape=(max_rows,)
    )
    ids_f = open(IDS_PATH, "w", encoding="utf-8")

    n_written = 0
    n_seen = 0
    n_dup = 0  # same 001 seen more than once across files
    seen_ids: set[str] = set()
    t_start = time.time()
    t_last = t_start

    try:
        for fi, path in enumerate(files, start=1):
            print(
                f"\n[{fi}/{len(files)}] {os.path.basename(path)}",
                flush=True,
            )
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    n_seen += 1

                    now = time.time()
                    if now - t_last >= PROGRESS_INTERVAL:
                        elapsed = now - t_start
                        rate = n_seen / elapsed if elapsed > 0 else 0
                        print(
                            f"  [{fmt_elapsed(elapsed)}] seen: {n_seen:,}  "
                            f"hits: {n_written:,}  dup: {n_dup:,}  "
                            f"rate: {rate:,.0f}/s",
                            flush=True,
                        )
                        t_last = now

                    # Cheap pre-filter: try to find lc_001 without full JSON parse.
                    # Embedding vectors are large; parsing every row is wasteful
                    # when only ~6% match. Find "lc_001" first.
                    idx = line.find('"lc_001"')
                    if idx < 0:
                        continue
                    # Find the value string right after.
                    colon = line.find(":", idx)
                    if colon < 0:
                        continue
                    quote_open = line.find('"', colon)
                    if quote_open < 0:
                        continue
                    quote_close = line.find('"', quote_open + 1)
                    if quote_close < 0:
                        continue
                    lc_001 = line[quote_open + 1:quote_close]

                    ddc = ddc_table.get(lc_001)
                    if ddc is None:
                        continue

                    if lc_001 in seen_ids:
                        n_dup += 1
                        continue

                    # Now parse the row for real to get the embedding.
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    emb = obj.get("embedding")
                    if not isinstance(emb, list) or len(emb) != DIM:
                        continue

                    vectors[n_written, :] = emb
                    ddcs[n_written] = ddc
                    ids_f.write(lc_001)
                    ids_f.write("\n")
                    seen_ids.add(lc_001)
                    n_written += 1

                    if n_written >= max_rows:
                        # Should not happen unless the DDC table grew under us.
                        print("WARN: reached pre-allocated capacity", flush=True)
                        break
    finally:
        ids_f.close()
        vectors.flush()
        ddcs.flush()
        del vectors
        del ddcs

    elapsed = time.time() - t_start
    print(
        f"\nScan complete in {fmt_elapsed(elapsed)}.  "
        f"seen: {n_seen:,}  hits: {n_written:,}  dup_skipped: {n_dup:,}",
        flush=True,
    )

    # Truncate memmaps to actual size by rewriting with the right shape header.
    print(f"\nTruncating arrays to {n_written:,} rows...", flush=True)
    final_vec_path = VECTORS_PATH + ".tmp"
    final_ddc_path = DDC_PATH + ".tmp"

    src_v = np.load(VECTORS_PATH, mmap_mode="r")
    src_d = np.load(DDC_PATH, mmap_mode="r")
    dst_v = np.lib.format.open_memmap(
        final_vec_path, mode="w+", dtype=np.float32, shape=(n_written, DIM)
    )
    dst_d = np.lib.format.open_memmap(
        final_ddc_path, mode="w+", dtype=np.float64, shape=(n_written,)
    )
    # Copy in chunks to keep memory bounded.
    chunk = 100_000
    for i in range(0, n_written, chunk):
        j = min(i + chunk, n_written)
        dst_v[i:j] = src_v[i:j]
        dst_d[i:j] = src_d[i:j]
    dst_v.flush()
    dst_d.flush()
    del src_v, src_d, dst_v, dst_d

    os.replace(final_vec_path, VECTORS_PATH)
    os.replace(final_ddc_path, DDC_PATH)

    print(f"\nWrote:")
    print(f"  {VECTORS_PATH}   ({n_written:,} x {DIM} float32)")
    print(f"  {DDC_PATH}       ({n_written:,} float64)")
    print(f"  {IDS_PATH}       ({n_written:,} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
