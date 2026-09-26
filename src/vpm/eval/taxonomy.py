"""Automatic classification of every failure into a named error class.

The brief asks us to *diagnose why*, and prose does not scale to a hundred
photographs. The colourway graph makes most of this mechanical: given the true
item, the retrieved item and the catalogue metadata, the class of a failure is
usually determined.

The distinction that carries the most information is **colourway confusion** --
the right model in the wrong colour. At SKU level that is a miss, but it is a
categorically different failure from returning an unrelated shoe, and lumping
them together is what turns an error analysis into a single uninformative number.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CLASSES = [
    "correct",
    "colourway_confusion",     # right style cluster, wrong SKU
    "same_brand_confusion",    # same brand, different style
    "silhouette_confusion",    # different brand, same article type (e.g. both sneakers)
    "catastrophic",            # unrelated: different article type
    "sku_absent",              # the true item is not in the index at all
]


@dataclass
class ItemMeta:
    brand: str
    article_type: str
    base_colour: str | None


def load_meta(items_path: Path) -> dict[int, ItemMeta]:
    meta: dict[int, ItemMeta] = {}
    with Path(items_path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            meta[r["product_id"]] = ItemMeta(
                brand=(r.get("brand") or "").strip().lower(),
                article_type=(r.get("article_type") or "").strip().lower(),
                base_colour=(r.get("base_colour") or None),
            )
    return meta


def classify(
    truth: int | None,
    predicted: int | None,
    meta: dict[int, ItemMeta],
    style_of: dict[int, int],
    indexed: set[int] | None = None,
) -> str:
    """Assign exactly one class. Order matters: earlier tests are more specific."""
    if truth is None or predicted is None:
        return "catastrophic"
    if predicted == truth:
        return "correct"
    if indexed is not None and truth not in indexed:
        # Not a retrieval failure at all -- the right answer was never available.
        return "sku_absent"
    if style_of.get(truth, truth) == style_of.get(predicted, predicted):
        return "colourway_confusion"

    mt, mp = meta.get(truth), meta.get(predicted)
    if mt is None or mp is None:
        return "catastrophic"
    if mt.brand and mt.brand == mp.brand:
        return "same_brand_confusion"
    if mt.article_type and mt.article_type == mp.article_type:
        return "silhouette_confusion"
    return "catastrophic"


def summarise(rows: list[dict]) -> dict:
    """Counts and shares per class, plus the share of errors each class represents."""
    counts = Counter(r["error_class"] for r in rows)
    n = max(len(rows), 1)
    n_err = max(sum(v for k, v in counts.items() if k != "correct"), 1)
    return {
        "n": len(rows),
        "counts": {c: counts.get(c, 0) for c in CLASSES},
        "share_of_all": {c: counts.get(c, 0) / n for c in CLASSES},
        "share_of_errors": {
            c: (counts.get(c, 0) / n_err) for c in CLASSES if c != "correct"
        },
    }


def to_markdown(summary: dict) -> str:
    L = ["### Error taxonomy\n",
         "| class | n | share of all | share of errors |", "|---|---|---|---|"]
    for c in CLASSES:
        n = summary["counts"][c]
        if n == 0 and c == "sku_absent":
            continue
        share_err = summary["share_of_errors"].get(c)
        L.append(f"| {c} | {n} | {summary['share_of_all'][c]:.3f} | "
                 f"{'—' if share_err is None else f'{share_err:.3f}'} |")
    L.append("")
    L.append("> `colourway_confusion` is the right model in the wrong colour. At SKU "
             "level it is a miss, but it is a categorically different failure from "
             "returning an unrelated shoe, and reporting one number for both hides "
             "which problem you actually have.")
    return "\n".join(L)
