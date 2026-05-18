#!/usr/bin/env python3.12
"""
Extract 3-digit DDC class labels from the Wikipedia "List of Dewey Decimal
classes" HTML snapshot.

Reads:  ddc-wikipedia.html
Writes: /Volumes/ImNotGlum/ddc_embedding/ddc_labels.tsv

Output is TSV with header `class\tlabel`, one row per 3-digit code that
Wikipedia documents (typically ~973 of the 1000 codes; the rest are gaps
that Dewey left unassigned).

Labels come from <li> items whose text starts with three digits. Top-level
headers (000, 100, ..., 900) and division headers (010, 020, ..., 990)
appear both as section headings and as list items, which the parser would
otherwise merge — so any second "NNN " seen inside a label truncates it.
Trailing footnote-style brackets (e.g. "[4]", "[moved to 017]") are stripped.
"""

import os
import re
import sys
from html.parser import HTMLParser

HTML_PATH = "ddc-wikipedia.html"
OUT_DIR = "/Volumes/ImNotGlum/ddc_embedding"
OUT_PATH = os.path.join(OUT_DIR, "ddc_labels.tsv")


class TextOnly(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def strip_html(fragment: str) -> str:
    t = TextOnly()
    t.feed(fragment)
    return re.sub(r"\s+", " ", t.text()).strip()


def parse_labels(html: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    for li in re.findall(r"<li[^>]*>(.*?)</li>", html, flags=re.DOTALL):
        text = strip_html(li)
        m = re.match(r"^(\d{3})\s+(.+)$", text)
        if not m:
            continue
        num = int(m.group(1))
        label = m.group(2).strip()

        # Wikipedia sometimes runs two adjacent list items together for
        # top-level/division codes. If the label contains a second " NNN ",
        # truncate before it.
        inner = re.search(r"\s\d{3}\s", label)
        if inner:
            label = label[: inner.start()].strip()

        # Drop trailing footnote brackets ("Foo[4]" -> "Foo", "Bar [moved to 017]" -> "Bar"),
        # but only if non-bracket text precedes the bracket — otherwise "[Unassigned]"
        # would be wiped out entirely.
        label = re.sub(r"(\S)\s*\[[^\]]+\]\s*$", r"\1", label).strip()

        if num not in labels:
            labels[num] = label
    return labels


def main() -> int:
    if not os.path.exists(HTML_PATH):
        print(f"Missing {HTML_PATH}", file=sys.stderr)
        return 1
    with open(HTML_PATH, encoding="utf-8") as f:
        html = f.read()

    labels = parse_labels(html)
    if not labels:
        print("No labels parsed.", file=sys.stderr)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("class\tlabel\n")
        for k in sorted(labels):
            f.write(f"{k:03d}\t{labels[k]}\n")

    print(f"Wrote {len(labels):,} labels to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
