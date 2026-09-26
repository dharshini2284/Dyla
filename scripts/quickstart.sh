#!/usr/bin/env bash
# Smallest end-to-end run that produces a working, queryable system.
#
# The full pipeline (scripts/run_all.sh) scrapes 12,000 products and downloads
# ~50,000 images, which takes hours. This builds a genuine but small catalogue so
# a reviewer can see the thing work, and every stage is the same code the full
# run uses -- nothing is stubbed.
set -euo pipefail

# Resolve helper scripts relative to THIS file, not the caller's cwd, so the
# script works from anywhere. Data still lands in ./data relative to cwd.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Index size. We SCRAPE more than we index on purpose: the distractor bank and
# the calibrator's negatives both come from items that are genuinely absent from
# the index, so a catalogue where everything is indexed leaves nothing to draw
# them from. At 400 scraped / 400 indexed only 26 items were absent and the
# calibrator refused to fit.
N_ITEMS=${N_ITEMS:-600}
N_SCRAPE=$(( N_ITEMS * 3 ))
PORT=${PORT:-8765}

echo "==> 1/6 harvesting footwear URLs from Myntra's sitemaps"
vpm scrape sitemaps --n-sitemaps 2 --out data/refs_quick.jsonl

echo "==> 2/6 fetching $N_SCRAPE product pages (indexing $N_ITEMS, rest become negatives)"
vpm scrape products --refs data/refs_quick.jsonl --out data/items.jsonl --limit "$N_SCRAPE"

echo "==> 3/6 downloading catalogue images"
vpm scrape images --items data/items.jsonl --images data/images --max-views 3

echo "==> 4/6 building the test set and index"
python "$HERE/make_synthetic_testset.py" --items data/items.jsonl --limit "$N_ITEMS" \
       --n-items 15 --per-item 2 --n-ooc 8
vpm index --backbone siglip2-base --whiten 256 --max-views 3 --limit "$N_ITEMS" \
          --items data/items.jsonl --images data/images \
          --exclude data/testset_synthetic/exclude_from_index.json --out data/index.npz

echo "==> 5/6 distractor bank + calibrator"
python "$HERE/build_distractors.py" --index data/index.npz --items data/items.jsonl \
       --images data/images --n "$N_ITEMS"
python "$HERE/fit_calibrator.py" --index data/index.npz --items data/items.jsonl \
       --images data/images --n-pos 200 --n-neg 150

echo "==> 6/6 evaluating"
vpm eval --index data/index.npz --items data/items.jsonl \
         --testset data/testset_synthetic --out reports/eval

cat <<EOF

  done.

  results   reports/eval/report.md
  failures  reports/eval/errors.html
  try it    vpm ui --items data/items.jsonl --port $PORT

  This is a small catalogue -- accuracy is not comparable to the numbers in
  REPORT.md, which come from a 3,000-item index. Use scripts/run_all.sh for that.
EOF
