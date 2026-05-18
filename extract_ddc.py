#!/usr/bin/env python3.12
"""
Extract Dewey Decimal Classification (MARC 082 $a) from LC bib records.

Streams /Volumes/ImNotGlum/lc_bibs/bibs.mrc record-by-record using the same
length-prefix read pattern as scripts/data/build_rag_docs.py, and writes
TSV rows of (001, ddc_numeric) to /Volumes/ImNotGlum/ddc_embedding/ddc.tsv.

Rules:
  - Only keep 082 fields with ind1 in {'0','1'} (full or abridged DDC).
  - For each record, take the FIRST $a that parses as a number after:
      * removing prime marks ('/')
      * stripping whitespace and trailing dots
  - A value is "numeric" if it matches ^\\d+(\\.\\d+)?$ after cleaning.
  - One row per 001. Records without a usable DDC are skipped.
"""

import re
import sys
import time

from pymarc import Record as MarcRecord

BIBS_FILE = "/Volumes/ImNotGlum/lc_bibs/bibs.mrc"
OUT_FILE = "/Volumes/ImNotGlum/ddc_embedding/ddc.tsv"
TOTAL_RECORDS = 20_329_074
PROGRESS_INTERVAL = 5.0

NUMERIC_RE = re.compile(r"^\d+(\.\d+)?$")


def clean_ddc(raw: str) -> str | None:
    if not raw:
        return None
    s = raw.replace("/", "").strip()
    s = s.rstrip(".")
    if NUMERIC_RE.match(s):
        return s
    return None


def extract_ddc(rec: MarcRecord) -> str | None:
    for fld in rec.get_fields("082"):
        ind1 = fld.indicators[0] if fld.indicators else " "
        if ind1 not in ("0", "1"):
            continue
        try:
            subs = [(s.code, s.value) for s in fld.subfields]
        except AttributeError:
            sf = fld.subfields
            subs = list(zip(sf[::2], sf[1::2]))
        for code, value in subs:
            if code != "a":
                continue
            cleaned = clean_ddc(value)
            if cleaned is not None:
                return cleaned
    return None


def fmt_elapsed(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


def fmt_rate(n: int, elapsed: float) -> str:
    if elapsed <= 0:
        return "  ?/s"
    return f"{n / elapsed:,.0f}/s"


def main() -> int:
    rec_count = 0
    written = 0
    no_ddc = 0
    bad_rec = 0
    t_start = time.time()
    t_last = t_start

    with open(BIBS_FILE, "rb") as fin, open(OUT_FILE, "w", encoding="utf-8") as fout:
        fout.write("001\tddc\n")
        while True:
            leader_len = fin.read(5)
            if len(leader_len) < 5:
                break
            try:
                rec_len = int(leader_len)
            except ValueError:
                break
            rest = fin.read(rec_len - 5)
            if len(rest) < rec_len - 5:
                break
            rec_count += 1
            raw = leader_len + rest

            now = time.time()
            if now - t_last >= PROGRESS_INTERVAL:
                elapsed = now - t_start
                pct = rec_count / TOTAL_RECORDS * 100
                print(
                    f"  [{fmt_elapsed(elapsed)}] {rec_count:>10,} ({pct:.1f}%)  "
                    f"written: {written:,}  no_ddc: {no_ddc:,}  "
                    f"bad_rec: {bad_rec:,}  rate: {fmt_rate(rec_count, elapsed)}",
                    flush=True,
                )
                t_last = now

            try:
                rec = MarcRecord(data=raw, to_unicode=True, force_utf8=True)
            except Exception:
                bad_rec += 1
                continue

            f001 = rec.get_fields("001")
            if not f001:
                continue
            lc_001 = f001[0].data.strip()
            if not lc_001:
                continue

            ddc = extract_ddc(rec)
            if ddc is None:
                no_ddc += 1
                continue

            fout.write(f"{lc_001}\t{ddc}\n")
            written += 1

    elapsed = time.time() - t_start
    print(
        f"\nDone in {fmt_elapsed(elapsed)}.  "
        f"records: {rec_count:,}  written: {written:,}  "
        f"no_ddc: {no_ddc:,}  bad_rec: {bad_rec:,}",
        flush=True,
    )
    print(f"Output: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
