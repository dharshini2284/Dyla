"""Encode a distractor bank from footwear guaranteed absent from the catalogue.

Selection is by STYLE CLUSTER, not by item: holding out an item whose colourway
twin remains indexed would put a near-duplicate of a "distractor" back into the
very index it is meant to be independent of.
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

from vpm.catalogue.clusters import style_clusters
from vpm.catalogue.trim import trim_to_square
from vpm.embed.backbone import EncodeConfig, build_backbone
from vpm.embed.encode import CatalogueEmbeddings
from vpm.index.distractors import build_from_embeddings
from vpm.index.pca import PCAWhitening
from vpm.scrape.download import image_path

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.npz"))
    ap.add_argument("--items", type=Path, default=Path("data/items.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/images"))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("data/distractors.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ce = CatalogueEmbeddings.load(args.index)
    bb = build_backbone(ce.backbone,
                        config=EncodeConfig(image_size=ce.image_size, pooling=ce.pooling))
    whitener = None
    wpath = args.index.with_suffix(".whiten.npz")
    if ce.whiten_dim and wpath.exists():
        whitener = PCAWhitening.load(wpath)

    indexed = set(ce.item_ids.tolist())
    style_of = style_clusters(args.items)
    indexed_styles = {style_of.get(p, p) for p in indexed}

    rows = [json.loads(l) for l in args.items.open() if l.strip()]
    absent = [r for r in rows
              if r["product_id"] not in indexed
              and style_of.get(r["product_id"], r["product_id"]) not in indexed_styles]
    print(f"index holds {len(indexed)} items / {len(indexed_styles)} styles; "
          f"{len(absent)} items are absent at STYLE level")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(absent)

    vecs, buf, used = [], [], 0
    for r in absent:
        if used >= args.n:
            break
        p = image_path(args.images, r["product_id"], 0)
        if not p.exists():
            continue
        try:
            img = Image.open(p)
            img.load()
        except Exception:
            continue
        buf.append(trim_to_square(img, size=bb.cfg.image_size)[0])
        used += 1
        if len(buf) >= args.batch:
            batch = torch.stack([bb.transform(i) for i in buf])
            vecs.append(bb.encode_tensor(batch).numpy())
            buf.clear()
            if used % 1000 == 0:
                print(f"  encoded {used}", flush=True)
    if buf:
        batch = torch.stack([bb.transform(i) for i in buf])
        vecs.append(bb.encode_tensor(batch).numpy())

    emb = np.concatenate(vecs).astype(np.float32) if vecs else np.zeros((0, bb.dim), np.float32)
    if whitener is not None and len(emb):
        emb = whitener.transform(emb)
    db = build_from_embeddings(emb)
    db.save(args.out)
    print(f"wrote {len(db)} distractor embeddings ({db.dim}d) -> {args.out}")


if __name__ == "__main__":
    main()
