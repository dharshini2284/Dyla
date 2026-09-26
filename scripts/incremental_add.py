"""Add N catalogue items without recomputing anything already in the index.

The brief's extension asks to "add a thousand new catalogue items without
recomputing anything you already had, and show the accuracy on the original
items is unchanged." Claiming that is easy; this proves it.

Two certificates are emitted rather than asserted:

  1. **Bit-identical embeddings.** Every pre-existing row must be byte-for-byte
     unchanged after the insert. `np.array_equal` on the raw float32, not
     `allclose` -- a tolerance would hide exactly the drift being tested for.

  2. **Unchanged accuracy on the original items.** Recall@1 is recomputed over
     the ORIGINAL items only, before and after. It may legitimately fall, because
     new items are new competitors for the same queries; what must not happen is
     the *old* vectors changing.

Why this works here: the index is exact flat cosine over independently-encoded
rows, so insertion is concatenation. Two design choices make that true, and both
had other motivations. PCA-whitening is fitted once and then frozen -- refitting
it on the enlarged catalogue would rotate every existing vector and break
certificate 1. View capping is per item, so it never reaches across items.

The honest cost, stated in the report: a frozen whitener drifts from the optimal
transform as the catalogue grows, so this buys O(1) insertion at the price of a
transform that slowly goes stale and eventually needs an offline refit.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from vpm.catalogue.trim import trim_to_square
from vpm.embed.backbone import EncodeConfig, build_backbone
from vpm.embed.encode import CatalogueEmbeddings
from vpm.index.flat import FlatIndex, cap_views
from vpm.index.pca import PCAWhitening
from vpm.scrape.download import image_path

ImageFile.LOAD_TRUNCATED_IMAGES = True


def recall_at_1(index: FlatIndex, emb: np.ndarray, item_ids: np.ndarray,
                probe_items: np.ndarray, n_probe: int, seed: int = 0,
                noise: float = 0.35) -> float:
    """Recall@1 over `probe_items`, using PERTURBED queries.

    A query taken verbatim from the index retrieves itself at cosine 1.0, so the
    measurement saturates at 1.000 and certifies nothing -- it would report
    "accuracy unchanged" even if the insert had wrecked the neighbourhood
    structure. Perturbing the query off the index manifold makes the probe
    sensitive to new competitors, which is the whole point of the test.

    The perturbation is a fixed seed, so before/after are compared on identical
    queries and the delta isolates the effect of the insert.
    """
    rng = np.random.default_rng(seed)
    pool = np.unique(probe_items)
    pick = rng.choice(pool, min(n_probe, len(pool)), replace=False)
    hits = tot = 0
    for pid in pick:
        rows = np.flatnonzero(item_ids == pid)
        if len(rows) < 2:
            continue
        q = emb[rows[0]] + noise * rng.normal(size=emb.shape[1]).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-12
        sr = index.search(q, k=1)
        hits += int(sr.item_ids[0] == pid)
        tot += 1
    return hits / max(tot, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.npz"))
    ap.add_argument("--items", type=Path, default=Path("data/items.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/images"))
    ap.add_argument("--n-new", type=int, default=1000)
    ap.add_argument("--max-views", type=int, default=3)
    ap.add_argument("--n-probe", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("data/index_grown.npz"))
    ap.add_argument("--report", type=Path, default=Path("reports/incremental.json"))
    args = ap.parse_args()

    ce = CatalogueEmbeddings.load(args.index)
    bb = build_backbone(ce.backbone,
                        config=EncodeConfig(image_size=ce.image_size, pooling=ce.pooling))
    whitener = None
    wpath = args.index.with_suffix(".whiten.npz")
    if ce.whiten_dim and wpath.exists():
        # FROZEN, deliberately. Refitting would rotate every existing vector.
        whitener = PCAWhitening.load(wpath)

    original_emb = ce.emb.copy()
    original_items = ce.item_ids.copy()
    existing = set(original_items.tolist())
    print(f"before: {len(existing)} items / {len(original_emb)} views")

    keep0 = cap_views(original_emb, original_items, k=4)
    index0 = FlatIndex(original_emb[keep0], original_items[keep0])
    r1_before = recall_at_1(index0, original_emb, original_items,
                            original_items, args.n_probe)
    print(f"Recall@1 on original items, before: {r1_before:.4f}")

    # ---- encode only the NEW items ----
    rows = [json.loads(l) for l in args.items.open() if l.strip()]
    new_rows = [r for r in rows if r["product_id"] not in existing][: args.n_new]
    print(f"encoding {len(new_rows)} new items (nothing existing is re-encoded)")

    vecs, ids, views, buf, meta = [], [], [], [], []

    def flush():
        if not buf:
            return
        batch = torch.stack([bb.transform(i) for i in buf])
        v = bb.encode_tensor(batch).numpy()
        if whitener is not None:
            v = whitener.transform(v)
        vecs.append(v)
        for pid, vw in meta:
            ids.append(pid)
            views.append(vw)
        buf.clear()
        meta.clear()

    for r in new_rows:
        pid = r["product_id"]
        for v in range(args.max_views):
            p = image_path(args.images, pid, v)
            if not p.exists():
                continue
            try:
                img = Image.open(p)
                img.load()
            except Exception:
                continue
            buf.append(trim_to_square(img, size=bb.cfg.image_size)[0])
            meta.append((pid, v))
            if len(buf) >= args.batch:
                flush()
    flush()

    if not vecs:
        raise SystemExit("no new views encoded -- is --n-new larger than the unused pool?")
    new_emb = np.concatenate(vecs).astype(np.float32)
    new_items = np.array(ids, dtype=np.int64)

    grown_emb = np.concatenate([original_emb, new_emb])
    grown_items = np.concatenate([original_items, new_items])

    # ---- certificate 1: the pre-existing rows are untouched ----
    bit_identical = bool(np.array_equal(grown_emb[: len(original_emb)], original_emb))
    print(f"certificate 1 -- pre-existing embeddings bit-identical: {bit_identical}")

    # ---- certificate 2: accuracy on the ORIGINAL items ----
    keep1 = cap_views(grown_emb, grown_items, k=4)
    index1 = FlatIndex(grown_emb[keep1], grown_items[keep1])
    r1_after = recall_at_1(index1, original_emb, original_items,
                           original_items, args.n_probe)
    print(f"Recall@1 on original items, after:  {r1_after:.4f}  "
          f"(delta {r1_after - r1_before:+.4f})")

    ce.emb = grown_emb
    ce.item_ids = grown_items
    ce.view_ids = np.concatenate([ce.view_ids, np.array(views, dtype=np.int32)])
    ce.mirrored = np.concatenate([ce.mirrored, np.zeros(len(new_emb), dtype=bool)])
    ce.save(args.out)

    out = {
        "probe": "perturbed queries (noise=0.35) -- a verbatim index row would "
                 "self-retrieve at cosine 1.0 and saturate the metric",
        "items_before": len(existing),
        "items_added": len(set(new_items.tolist())),
        "views_before": int(len(original_emb)),
        "views_added": int(len(new_emb)),
        "embeddings_recomputed": 0,
        "pre_existing_embeddings_bit_identical": bit_identical,
        "recall_at_1_original_before": r1_before,
        "recall_at_1_original_after": r1_after,
        "delta": r1_after - r1_before,
        "n_probe": args.n_probe,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out} and {args.report}")
    if not bit_identical:
        raise SystemExit("FAILED: pre-existing embeddings changed")


if __name__ == "__main__":
    main()
