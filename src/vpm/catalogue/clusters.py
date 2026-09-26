"""Style clusters over the colourway graph.

Two jobs, both load-bearing:

1. **Ground truth is ambiguous at SKU level.** "Puma Softride, Black/Red" and
   "Puma Softride, Triple Black" are separate SKUs whose photos differ in a
   panel or two. Asking a matcher to pick the right one from a dark phone photo
   is sometimes genuinely unanswerable, so accuracy is reported at BOTH SKU and
   style level, and "correct style, wrong colourway" becomes its own error class
   rather than an unexplained miss.

2. **Naive item hold-out does not produce open-set negatives.** Measured on our
   catalogue: 36% of individually held-out items still had a colourway variant
   sitting in the index, so the "absent" query had a near-twin to match against.
   That alone drove bake-off AUROC below 0.5 -- worse than chance. Genuine
   negatives require holding out the whole cluster.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Colour words that appear in Myntra product names. Stripping them from a name
# leaves a model key, which links colourway siblings the `colours` field misses.
_COLOUR_WORDS = set("""
black white grey gray navy blue red green yellow orange pink purple brown beige
tan charcoal olive maroon teal cream silver gold bronze burgundy khaki mustard
coffee rust peach lavender turquoise magenta ivory sand stone wine multi
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def model_key(brand: str, name: str) -> tuple[str, str]:
    """(brand, name-with-colour-words-removed).

    "Sparx Men Charcoal Mesh Running Shoes" and "Sparx Men Navy Blue Mesh Running
    Shoes" collapse to the same key, which is what makes them recognisable as one
    model when the catalogue's own colour graph does not link them.
    """
    toks = [t for t in _TOKEN.findall((name or "").lower()) if t not in _COLOUR_WORDS]
    return ((brand or "").strip().lower(), " ".join(toks))


class _DSU:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:       # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def style_clusters(
    items_path: Path,
    restrict: set[int] | None = None,
    use_name_key: bool = True,
) -> dict[int, int]:
    """Map product_id -> style_id (the cluster's smallest product id).

    Edges come from two sources:

    1. the catalogue's own `colours` graph, which lists the same model in other
       colourways; and
    2. a **name-derived model key**, because that graph is incomplete.

    (2) is not defensive programming, it is a measured hole. "Sparx Men Charcoal
    Mesh Running Shoes" carries an EMPTY `colours` list while its Navy Blue
    sibling exists separately in the catalogue. Across 36,506 items the colour
    graph misses **3,563 links spanning 9.8% of items** -- 3,457 model names are
    split across clusters that should be one. This surfaced as a false accept:
    an item held out as "out of catalogue" had its own colourway twin sitting in
    the index, because nothing linked them.

    That matters twice over. Style-level accuracy is understated, and -- worse --
    open-set evaluation is corrupted, since "absent" items are not absent when a
    near-identical sibling remains indexed.

    `use_name_key=False` reproduces the colour-graph-only behaviour for ablation.
    """
    rows = []
    with items_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    present = {r["product_id"] for r in rows}
    if restrict is not None:
        present &= restrict

    dsu = _DSU()
    by_model: dict[tuple[str, str], int] = {}
    for r in rows:
        pid = r["product_id"]
        if pid not in present:
            continue
        dsu.find(pid)
        for v in r.get("colour_variants", []):
            if v in present:
                dsu.union(pid, v)
        if use_name_key:
            key = model_key(r.get("brand", ""), r.get("name", ""))
            if key[1]:
                if key in by_model:
                    dsu.union(pid, by_model[key])
                else:
                    by_model[key] = pid

    # Name each cluster by its smallest member so ids are stable across runs.
    groups: dict[int, list[int]] = {}
    for pid in present:
        groups.setdefault(dsu.find(pid), []).append(pid)
    return {pid: min(members) for members in groups.values() for pid in members}
