"""Re-download the real-world photo set listed in data/real_world/metadata.csv.

The images are publicly sourced, user-uploaded Wikimedia Commons photos (not
taken by the author of this repo). They are gitignored; the metadata is not, so
a clean clone can rebuild the image folder byte-for-byte: every row records the
exact URL fetched and the SHA-256 of the file that was labelled.

Politeness is deliberate, not incidental: one request per second, a descriptive
User-Agent (Wikimedia's policy requires one), backoff on 429/503, and only
upload.wikimedia.org is contacted -- the file hosts robots.txt allows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = "DylaFootwearEval/0.1 (visual product matcher test set; low-speed; see README)"


def fetch(url: str, tries: int = 4) -> bytes | None:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(10 * (attempt + 1))
                continue
            print(f"  HTTP {e.code}: {url}", file=sys.stderr)
            return None
        except urllib.error.URLError as e:
            print(f"  {e.reason}: {url}", file=sys.stderr)
            time.sleep(5)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/real_world"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    rows = list(csv.DictReader((args.root / "metadata.csv").open(newline="", encoding="utf-8")))
    got = skipped = failed = mismatched = 0
    for row in rows:
        dest = args.root / row["image_path"]
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == row["sha256"]:
            skipped += 1
            continue
        if not row["download_url"].startswith("https://upload.wikimedia.org/"):
            print(f"  refusing non-Commons URL for {row['photo_id']}", file=sys.stderr)
            failed += 1
            continue
        data = fetch(row["download_url"])
        time.sleep(args.delay)
        if data is None:
            failed += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            # Commons re-renders thumbnails occasionally; the photo is the same, the bytes may not be.
            mismatched += 1
        got += 1

    print(f"{len(rows)} rows: {got} downloaded, {skipped} already present, "
          f"{failed} failed, {mismatched} downloaded with a different hash")


if __name__ == "__main__":
    main()
