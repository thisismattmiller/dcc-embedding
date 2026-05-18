#!/usr/bin/env python3.12
"""
Phase 4: Render PCA and UMAP centroid maps as zoomable SVG embedded in a
single self-contained HTML page.

Inputs: Phase 2 centroids + Phase 2 class_ids/counts + Wikipedia labels.
Output: ddc_web/index.html plus ddc_web/{pca,umap}.svg
Open ddc_web/index.html in any modern browser. Mouse-wheel = zoom,
drag = pan. Labels are real SVG <text> nodes so they stay crisp at any
zoom level and are selectable / searchable with Ctrl-F.
"""

import html
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA

IN_DIR = "/Volumes/ImNotGlum/ddc_embedding"
CENTROIDS_PATH = os.path.join(IN_DIR, "centroids.f32.npy")
CLASS_IDS_PATH = os.path.join(IN_DIR, "class_ids.i32.npy")
COUNTS_PATH = os.path.join(IN_DIR, "counts.i64.npy")
LABELS_PATH = os.path.join(IN_DIR, "ddc_labels.tsv")

OUT_DIR = "docs"
MIN_COUNT = 20

SVG_W = 4000
SVG_H = 4000
PAD = 200


def load_labels(path: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            try:
                out[int(parts[0])] = parts[1]
            except ValueError:
                continue
    return out


def turbo_hex(t: float) -> str:
    """Approximate Google's turbo colormap. t in [0,1] -> #RRGGBB."""
    t = max(0.0, min(1.0, t))
    # Polynomial fit from the turbo colormap paper.
    r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))))
    g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))))
    b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814.0 - t * (22569.18 - t * 6838.66))))
    r = int(max(0, min(255, r)))
    g = int(max(0, min(255, g)))
    b = int(max(0, min(255, b)))
    return f"#{r:02x}{g:02x}{b:02x}"


def normalize_coords(coords: np.ndarray, w: int, h: int, pad: int) -> np.ndarray:
    xs, ys = coords[:, 0], coords[:, 1]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    sx = (w - 2 * pad) / max(x1 - x0, 1e-9)
    sy = (h - 2 * pad) / max(y1 - y0, 1e-9)
    s = min(sx, sy)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    out = np.empty_like(coords)
    out[:, 0] = (xs - cx) * s + w / 2
    out[:, 1] = (cy - ys) * s + h / 2  # flip Y for SVG screen coords
    return out


def assign_label_tiers(coords, counts, n_tiers=4):
    """Greedy collision-aware label tiering.

    Tier 0 = labels that should always show (most important, never collide
    with each other at default zoom). Tier 1 reveals at moderate zoom,
    tier 2 closer in, tier 3 only at max zoom.

    Importance = book count (log). For each tier, greedily pick points whose
    label box doesn't collide with any already-picked point in the SAME or
    EARLIER tier, using a tier-dependent collision radius (large at tier 0,
    small at tier 3).
    """
    n = len(coords)
    importance = np.log1p(counts.astype(np.float64))
    order = np.argsort(-importance)  # descending

    tiers = np.full(n, n_tiers - 1, dtype=np.int32)  # default = deepest tier
    placed_xy = []  # list of (x, y, tier)

    # Collision radii in SVG units, shrinking per tier.
    # Tuned for SVG_W = 4000. Tier 0 = ~280px gap, tier 3 = ~40px gap.
    radii = [280.0, 140.0, 75.0, 40.0]

    for idx in order:
        x, y = coords[idx]
        for tier in range(n_tiers):
            r = radii[tier]
            ok = True
            for px, py, _ in placed_xy:
                if (px - x) ** 2 + (py - y) ** 2 < r * r:
                    ok = False
                    break
            if ok:
                tiers[idx] = tier
                placed_xy.append((x, y, tier))
                break
        else:
            tiers[idx] = n_tiers - 1
            placed_xy.append((x, y, n_tiers - 1))
    return tiers


def render_svg(coords, class_ids, counts, labels, title, svg_id, n_tiers=4):
    tiers = assign_label_tiers(coords, counts, n_tiers=n_tiers)

    # Point radii: log-scaled by count so 813 (188k) is visibly bigger than
    # a 20-book class. Clamp to a sensible visual range.
    log_counts = np.log1p(counts.astype(np.float64))
    lo, hi = float(log_counts.min()), float(log_counts.max())
    span = max(hi - lo, 1e-9)
    radii = 4.0 + 10.0 * (log_counts - lo) / span  # 4..14 px

    parts = [
        f'<svg id="{svg_id}" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {SVG_W} {SVG_H}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'font-family="system-ui, sans-serif">',
        f'<rect width="{SVG_W}" height="{SVG_H}" fill="#0b0b0b"/>',
        f'<text x="{SVG_W//2}" y="60" text-anchor="middle" font-size="40" '
        f'fill="#eee">{html.escape(title)}</text>',
    ]

    # Line segments between consecutive centroids. Wrapped in a group so we
    # can dim it from the host page (the spaghetti is informative when
    # zoomed out, distracting when zoomed in).
    parts.append('<g class="ddc-edges" stroke-width="1.6" stroke-opacity="0.45" fill="none">')
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        mid = (int(class_ids[i]) + int(class_ids[i + 1])) / 2
        color = turbo_hex(mid / 999.0)
        parts.append(
            f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" '
            f'x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{color}"/>'
        )
    parts.append("</g>")

    # Points. Always drawn, sized by count. Each carries data-* attributes
    # the host page reads to populate the custom hover tooltip.
    parts.append('<g class="ddc-points">')
    for i, (x, y) in enumerate(coords):
        cid = int(class_ids[i])
        color = turbo_hex(cid / 999.0)
        label = labels.get(cid, "")
        r = radii[i]
        parts.append(
            f'<circle class="ddc-pt" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
            f'stroke="#000" stroke-width="0.5" '
            f'data-cid="{cid:03d}" '
            f'data-label="{html.escape(label, quote=True)}" '
            f'data-count="{int(counts[i])}"/>'
        )
    parts.append("</g>")

    # Labels, in tier groups (0..n_tiers-1). Each tier has its own opacity
    # toggle in the host page so we can fade lower tiers in as the user zooms.
    for tier in range(n_tiers):
        # Font shrinks slightly for deeper tiers to make zoomed-in detail
        # less overwhelming.
        num_size = 22 - tier * 4   # 22, 18, 14, 10
        lab_size = 18 - tier * 4   # 18, 14, 10, 6  (clamped below)
        lab_size = max(lab_size, 6)
        parts.append(f'<g class="ddc-labels ddc-tier-{tier}" pointer-events="none">')
        for i, (x, y) in enumerate(coords):
            if int(tiers[i]) != tier:
                continue
            cid = int(class_ids[i])
            label = labels.get(cid, "")
            short = label if len(label) <= 60 else label[:57] + "..."
            r = radii[i]
            parts.append(
                f'<text x="{x + r + 4:.1f}" y="{y - 2:.1f}" '
                f'font-size="{num_size}" fill="#fff" font-weight="700">'
                f'{cid:03d}</text>'
                f'<text x="{x + r + 4:.1f}" y="{y + lab_size + 2:.1f}" '
                f'font-size="{lab_size}" fill="#ddd">{html.escape(short)}</text>'
            )
        parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def render_html(out_path, pca_svg, umap_svg):
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DDC curve through embedding space</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #0b0b0b; color: #eee;
               font-family: system-ui, sans-serif; }}
  header {{ padding: 14px 20px; border-bottom: 1px solid #222;
           display: flex; gap: 24px; align-items: baseline; }}
  header h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
  header nav a {{ color: #8af; text-decoration: none; margin-right: 16px;
                 font-size: 14px; cursor: pointer; }}
  header nav a.active {{ color: #fff; border-bottom: 2px solid #fff; }}
  header .hint {{ color: #888; font-size: 13px; margin-left: auto; }}
  .stage {{ width: 100vw; height: calc(100vh - 52px);
           position: relative; overflow: hidden; }}
  .panel {{ position: absolute; inset: 0; display: none; }}
  .panel.active {{ display: block; }}
  .panel svg {{ width: 100%; height: 100%; display: block; }}

  /* Zoom-aware visibility. Default: only tier 0 labels and the edges show.
     As .zoom-1, .zoom-2, .zoom-3 are added to <body> by JS, deeper tiers
     fade in and the edges fade out. */
  .ddc-tier-0 {{ opacity: 1; }}
  .ddc-tier-1, .ddc-tier-2, .ddc-tier-3 {{ opacity: 0; transition: opacity 120ms; }}
  body.zoom-1 .ddc-tier-1 {{ opacity: 1; }}
  body.zoom-2 .ddc-tier-1, body.zoom-2 .ddc-tier-2 {{ opacity: 1; }}
  body.zoom-3 .ddc-tier-1, body.zoom-3 .ddc-tier-2, body.zoom-3 .ddc-tier-3 {{ opacity: 1; }}
  .ddc-edges {{ opacity: 0.55; transition: opacity 150ms; }}
  body.zoom-2 .ddc-edges {{ opacity: 0.25; }}
  body.zoom-3 .ddc-edges {{ opacity: 0.10; }}

  /* Custom hover tooltip. */
  .ddc-pt {{ cursor: pointer; }}
  .ddc-pt:hover {{ stroke: #fff; stroke-width: 2; }}
  #tooltip {{
    position: fixed; pointer-events: none; z-index: 1000;
    background: rgba(20,20,22,0.96); color: #fff;
    border: 1px solid #444; border-radius: 6px;
    padding: 8px 10px; font-size: 13px; line-height: 1.35;
    box-shadow: 0 4px 14px rgba(0,0,0,0.6);
    max-width: 360px; opacity: 0; transition: opacity 80ms;
  }}
  #tooltip.visible {{ opacity: 1; }}
  #tooltip .tt-cid {{ color: #8af; font-weight: 700; font-size: 14px; }}
  #tooltip .tt-label {{ margin-top: 2px; }}
  #tooltip .tt-count {{ margin-top: 4px; color: #999; font-size: 12px; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
</head>
<body>
<header>
  <h1>Dewey Decimal Classes through Embedding Space</h1>
  <nav>
    <a id="link-pca"  class="active">PCA</a>
    <a id="link-umap">UMAP</a>
  </nav>
  <div class="hint">Scroll to zoom · drag to pan · hover a node for full label · Ctrl/⌘-F to search labels</div>
</header>
<div class="stage">
  <div id="panel-pca"  class="panel active">{pca_svg}</div>
  <div id="panel-umap" class="panel">{umap_svg}</div>
</div>
<div id="tooltip"></div>
<script>
  const panZoom = {{}};
  function applyZoomLevel(z) {{
    // Thresholds in pan-zoom's "zoom" units (1.0 = initial fit).
    // Tuned so tier 1 reveals at ~2x, tier 2 at ~5x, tier 3 at ~15x.
    document.body.classList.toggle('zoom-1', z >= 2.0);
    document.body.classList.toggle('zoom-2', z >= 5.0);
    document.body.classList.toggle('zoom-3', z >= 15.0);
  }}
  function ensure(which) {{
    if (panZoom[which]) return;
    panZoom[which] = svgPanZoom('#svg-' + which, {{
      zoomEnabled: true,
      controlIconsEnabled: true,
      fit: true,
      center: true,
      minZoom: 0.5,
      maxZoom: 80,
      zoomScaleSensitivity: 0.35,
      onZoom: applyZoomLevel,
    }});
    applyZoomLevel(panZoom[which].getZoom());
  }}
  function show(which) {{
    ['pca', 'umap'].forEach(t => {{
      document.getElementById('panel-' + t).classList.toggle('active', t === which);
      document.getElementById('link-' + t).classList.toggle('active', t === which);
    }});
    ensure(which);
    // svg-pan-zoom needs a resize kick after the panel becomes visible.
    requestAnimationFrame(() => {{
      panZoom[which].resize();
      panZoom[which].fit();
      panZoom[which].center();
    }});
  }}
  ['pca', 'umap'].forEach(t => {{
    document.getElementById('link-' + t).addEventListener('click', e => {{
      e.preventDefault();
      history.replaceState(null, '', '#' + t);
      show(t);
    }});
  }});
  show(location.hash === '#umap' ? 'umap' : 'pca');

  // Custom hover tooltip. One listener on the stage handles all points
  // across both panels via event delegation; cheaper than attaching to 925
  // circles per panel.
  const tooltip = document.getElementById('tooltip');
  let activePt = null;
  function fmtNum(n) {{ return Number(n).toLocaleString(); }}
  function showTip(pt, ev) {{
    const cid = pt.getAttribute('data-cid');
    const label = pt.getAttribute('data-label') || '(no label)';
    const count = pt.getAttribute('data-count');
    tooltip.innerHTML =
      '<div class="tt-cid">' + cid + '</div>' +
      '<div class="tt-label">' + label + '</div>' +
      '<div class="tt-count">' + fmtNum(count) + ' books</div>';
    tooltip.classList.add('visible');
    moveTip(ev);
  }}
  function moveTip(ev) {{
    const pad = 14;
    let x = ev.clientX + pad;
    let y = ev.clientY + pad;
    const w = tooltip.offsetWidth, h = tooltip.offsetHeight;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }}
  function hideTip() {{
    activePt = null;
    tooltip.classList.remove('visible');
  }}
  document.querySelector('.stage').addEventListener('mouseover', e => {{
    const pt = e.target.closest('.ddc-pt');
    if (!pt) return;
    activePt = pt;
    showTip(pt, e);
  }});
  document.querySelector('.stage').addEventListener('mousemove', e => {{
    if (!activePt) return;
    // If the cursor moved off the active point, hide.
    if (!e.target.closest('.ddc-pt')) {{ hideTip(); return; }}
    moveTip(e);
  }});
  document.querySelector('.stage').addEventListener('mouseleave', hideTip);
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"  wrote {out_path}  ({os.path.getsize(out_path)//1024} KB)", flush=True)


def main() -> int:
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading centroids...", flush=True)
    centroids = np.load(CENTROIDS_PATH)
    class_ids = np.load(CLASS_IDS_PATH)
    counts = np.load(COUNTS_PATH)
    labels = load_labels(LABELS_PATH)

    keep = counts >= MIN_COUNT
    centroids = centroids[keep]
    class_ids = class_ids[keep]
    counts = counts[keep]
    order = np.argsort(class_ids)
    centroids = centroids[order]
    class_ids = class_ids[order]
    counts = counts[order]
    print(f"  {len(class_ids)} classes after >= {MIN_COUNT}-book filter", flush=True)

    print("\nFitting PCA...", flush=True)
    pca = PCA(n_components=2, random_state=0)
    pca_coords = pca.fit_transform(centroids)
    print(
        f"  explained: {pca.explained_variance_ratio_[0]*100:.1f}% + "
        f"{pca.explained_variance_ratio_[1]*100:.1f}%",
        flush=True,
    )
    pca_screen = normalize_coords(pca_coords, SVG_W, SVG_H, PAD)
    pca_svg = render_svg(pca_screen, class_ids, counts, labels,
                         "DDC centroids — PCA", "svg-pca")

    print("\nFitting UMAP (cosine)...", flush=True)
    import umap
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=0,
    )
    umap_coords = reducer.fit_transform(centroids)
    umap_screen = normalize_coords(np.asarray(umap_coords), SVG_W, SVG_H, PAD)
    umap_svg = render_svg(umap_screen, class_ids, counts, labels,
                          "DDC centroids — UMAP (cosine)", "svg-umap")

    render_html(os.path.join(OUT_DIR, "index.html"), pca_svg, umap_svg)

    print(f"\nDone in {time.time()-t0:.1f}s.")
    print(f"\nOpen: {os.path.abspath(os.path.join(OUT_DIR, 'index.html'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
