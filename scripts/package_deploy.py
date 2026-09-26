"""Package a deployable bundle: everything the UI needs, nothing it does not.

The working tree carries ~30 GB of full-resolution multi-view catalogue images,
which exist for experiments -- re-encoding at other resolutions, the incremental
test, the bake-off. A deployment needs exactly one 200px thumbnail per indexed
item (measured: 3.9 KB each) and the ~14 MB of index artefacts.

Output is self-contained and small enough for any free tier that can hold the
process in memory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vpm.embed.encode import CatalogueEmbeddings
from vpm.scrape.download import image_path

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Only the fields the UI actually renders. items_tier1.jsonl is 48 MB, mostly
# image URLs and variant graphs that a served page never reads.
KEEP = ("product_id", "brand", "name", "base_colour", "article_type")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index.npz"))
    ap.add_argument("--items", type=Path, default=Path("data/items_tier1.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/images"))
    ap.add_argument("--calibrator", type=Path, default=Path("data/calibrator"))
    ap.add_argument("--distractors", type=Path, default=Path("data/distractors.npz"))
    ap.add_argument("--out", type=Path, default=Path("deploy/bundle"))
    ap.add_argument("--thumb", type=int, default=200)
    ap.add_argument("--quality", type=int, default=78)
    args = ap.parse_args()

    ce = CatalogueEmbeddings.load(args.index)
    wanted = sorted(set(int(i) for i in ce.item_ids.tolist()))
    print(f"index holds {len(wanted)} items")

    out = args.out
    (out / "data").mkdir(parents=True, exist_ok=True)

    for src in (args.index, args.index.with_suffix(".whiten.npz"), args.distractors):
        if src.exists():
            shutil.copy2(src, out / "data" / src.name)
            print(f"  copied {src.name} ({src.stat().st_size/1e6:.1f} MB)")
    if args.calibrator.exists():
        shutil.copytree(args.calibrator, out / "data" / "calibrator", dirs_exist_ok=True)
        print("  copied calibrator/")

    # slim metadata
    meta_path = out / "data" / "items.jsonl"
    kept = 0
    want = set(wanted)
    with args.items.open(encoding="utf-8") as fh, meta_path.open("w", encoding="utf-8") as w:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["product_id"] in want:
                w.write(json.dumps({k: r.get(k) for k in KEEP}) + "\n")
                kept += 1
    print(f"  wrote items.jsonl: {kept} items ({meta_path.stat().st_size/1e6:.1f} MB)")

    # thumbnails, in the shard layout the server already expects
    troot = out / "data" / "images"
    made = missing = 0
    for pid in wanted:
        src = image_path(args.images, pid, 0)
        if not src.exists():
            missing += 1
            continue
        dst = troot / f"{pid % 100:02d}" / str(pid) / "0.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            im = Image.open(src).convert("RGB")
            im.thumbnail((args.thumb, args.thumb), Image.LANCZOS)
            im.save(dst, "JPEG", quality=args.quality, optimize=True)
            made += 1
        except Exception:
            missing += 1
        if made and made % 1000 == 0:
            print(f"  thumbnails {made}/{len(wanted)}", flush=True)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n  {made} thumbnails ({missing} missing)")
    print(f"  bundle: {out}  —  {size/1e6:.1f} MB total")
    print("\n  the ~400 MB backbone is NOT bundled; it downloads from HuggingFace at boot")
    print("  (bake it into the image with HF_HOME set if cold start matters)")


if __name__ == "__main__":
    main()
