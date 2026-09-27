# Visual Product Matcher — footwear

Photograph a shoe, get the matching catalogue SKU back, ranked, with a
confidence score that means something — and a refusal when the shoe is not in
the catalogue at all.

Catalogue: **footwear scraped from Myntra**. A 345,331-item footwear pool was
harvested from the public sitemaps; the working catalogue is a multi-view subset
of it.

- **[REPORT.md](REPORT.md)** — the write-up: what was measured, what was rejected, and what does not work.
- **[DECISIONS.md](DECISIONS.md)** — the decision log, one entry per decision with the evidence that moved it.
- **[logs/SESSION.md](logs/SESSION.md)** — how the work actually happened, including the two places the tool's recommendation was overruled with measurement. Raw run logs in `logs/runs/`.

---

## For reviewers — running this yourself

**One command, about seven minutes:**

```bash
git clone https://github.com/dharshini2284/Dyla && cd Dyla
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
./scripts/quickstart.sh          # ~7 min: scrapes, indexes, calibrates, evaluates
vpm ui --items data/items.jsonl  # then open http://127.0.0.1:8765/ui
```

Timed from a clean clone: **398 s (6.6 min)** for a 400-item index, producing
hard R@1 0.708, refusal AUROC 0.931, and a full `reports/eval/report.md` plus the
visual failure grid in `reports/eval/errors.html`.

Nothing in that path is stubbed — it is the same code the full run uses, on a
smaller catalogue. `./scripts/run_all.sh` builds the 3,000-item index the numbers
in `REPORT.md` come from, and takes hours because it scrapes and downloads
roughly fifty thousand images.

The derived artefacts (index, thumbnails, calibrator) are **not** in the repo —
they are regenerable, and the full-resolution image set is ~30 GB. The quickstart
exists so a clean clone is genuinely runnable rather than nominally so.

**Or as a container**, if you would rather not build a Python environment:

```bash
python scripts/package_deploy.py && docker build -t vpm-ui .
docker run --rm -p 8765:8765 vpm-ui
```

### A note on links

`http://127.0.0.1:8765` is **loopback only** — it is not reachable by anyone
else, including over a local network. There is no authentication on this server
and the catalogue is scraped third-party product data, so it is deliberately not
something to leave on a public URL. For a live walkthrough, a temporary tunnel
(`cloudflared tunnel --url http://localhost:8765`) gives a URL that dies when you
close it; for anything longer-lived see `deploy/README.md`.

## Quick start

Requires Python 3.11+. On Apple Silicon, PyTorch MPS is used automatically; the
code falls back to CPU everywhere.

```bash
git clone https://github.com/dharshini2284/Dyla.git && cd Dyla
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

That installs the `vpm` CLI. To reproduce from nothing:

```bash
# 1. Harvest footwear product URLs from Myntra's sitemaps (~2 min, no page fetches)
vpm scrape sitemaps --out data/refs_footwear.jsonl

# 2. Fetch product pages -> catalogue records (rate-limited; ~10 items/s)
vpm scrape products --refs data/refs_footwear.jsonl --out data/items.jsonl --limit 12000

# 3. Download catalogue images (resumable; re-run to pick up stragglers)
vpm scrape images --items data/items.jsonl --images data/images --max-views 4

# 4. Build a synthetic stand-in test set (see REPORT.md §Part B for why)
python scripts/make_synthetic_testset.py --n-items 40 --per-item 3 --n-ooc 20

# 5. Encode the catalogue into a whitened index (~15 min on an M3)
vpm index --backbone siglip2-base --whiten 256 --max-views 4 \
          --items data/items.jsonl --images data/images \
          --exclude data/testset_synthetic/exclude_from_index.json \
          --out data/index.npz

# 7. Match a single photo
vpm query path/to/photo.jpg --index data/index.npz --items data/items.jsonl

# 11. Latency breakdown for one lookup
vpm bench --index data/index.npz --iters 25

# 8. Build the held-out distractor bank -- footwear absent from the catalogue at
#    STYLE level, used to normalise confidence scores (see REPORT.md section 6)
python scripts/build_distractors.py --index data/index.npz --items data/items.jsonl

# 9. Fit the open-set calibrator and the conformal refuser
python scripts/fit_calibrator.py --index data/index.npz --items data/items.jsonl

# 10. Full evaluation: per-condition accuracy, marginal effects, error taxonomy,
#     refusal, gap analysis
vpm eval --index data/index.npz --items data/items.jsonl \
         --testset data/testset_synthetic --out reports/eval
```

Every table in `REPORT.md` comes from step 10. Nothing is hand-copied.

`vpm eval` also writes **`reports/eval/errors.html`** — a self-contained page
showing every failure with the query, the answer returned and the answer
expected, side by side, plus the per-condition and marginal-effect tables. It is
the one thing a results table cannot do: show *why* a case was hard. Thumbnails
are embedded, so the file travels as a single artefact.

### Interactive matcher

```bash
vpm ui --items data/items.jsonl        # then open http://127.0.0.1:8765/ui
```

Drop a photo and see the whole decision: the ranked top-5 with catalogue
thumbnails and scores, the calibrated confidence, the conformal prediction set,
an explicit **NO MATCH** badge when it refuses — with the reason stated ("a bank
of shoes known to be absent matched this photo better than the catalogue's best
guess") — a per-stage latency breakdown, and the patch-norm saliency overlay.

Stdlib `http.server` only; no Streamlit, no npm, nothing added to the lock file.
The saliency panel is shown as a **diagnostic**, not as part of the pipeline —
localisation is off by default (see `REPORT.md` §7).

### Matching a photo, and refusing one

A shoe that **is** in the catalogue:

```bash
vpm query data/testset_synthetic/hard/11198964_0.jpg --items data/items.jsonl
```

```
  1. 0.7629  11198964  Puma — Puma Men Charcoal Grey Jigsaw Sneakers
  2. 0.3947  20315022  Puma — Puma Men Grey Leather Running Shoes
  ...
confidence: p(top-1 correct and in catalogue) = 1.000
decision:   ACCEPT -- conformal set of 5 at alpha=0.1
            best distractor similarity 0.335 vs top-1 0.763
```

A shoe that is **not**:

```bash
vpm query data/testset_synthetic/ooc/13459372.jpg --items data/items.jsonl
```

```
  1. 0.5467  11334966  Puma — Puma Men Navy Blue Dryflex Sneakers
confidence: p(top-1 correct and in catalogue) = 0.024
decision:   NO MATCH -- refused (empty conformal set at alpha=0.1)
            best distractor similarity 0.607 vs top-1 0.547
```

Note what actually drove that refusal. The system still found a plausible-looking
Puma sneaker at 0.547 — a raw-similarity threshold would very likely have
accepted it. It refused because **a bank of shoes that are definitely not in the
catalogue matched the photo better (0.607) than the catalogue's best guess did
(0.547)**. That is the likelihood-ratio question from `REPORT.md` §6: *how much
better does the catalogue explain this photo than a generic pile of shoes does?*

To run the whole out-of-catalogue set:

```bash
for f in data/testset_synthetic/ooc/*.jpg; do
  vpm query "$f" --items data/items.jsonl | grep -E "^decision|best distractor"
done
```

**And it is not perfect — four of five refuse, one does not:**

| photo | top-1 | best distractor | p | decision | |
|---|---|---|---|---|---|
| 13459372 | 0.547 | **0.607** | 0.024 | refused | correct |
| 10937062 | 0.518 | 0.488 | 0.071 | refused | correct |
| 11168228 | 0.344 | 0.304 | 0.079 | refused | correct |
| 18653640 | 0.325 | 0.273 | 0.024 | refused | correct |
| 19896550 | 0.534 | 0.381 | 0.778 | **accepted** | **wrong** |

Compare rows 1 and 5: **top-1 similarity is almost identical (0.547 vs 0.534) and
the decisions are opposite.** Raw similarity carries essentially no information
here — the decision comes entirely from the distractor comparison, which is right
four times and wrong once. That is the measured FAR of 0.250, and with 20
negatives it is one photo wide either way.

(Row 3, `11168228`, is the photo that exposed the incomplete colourway graph in
`REPORT.md` §4.5. It was a false accept before that fix and refuses correctly
now.)

### Part B — the hand-shot test set

The 100 adversarial photographs are **not** in this repo; they require items the
author physically owns (see *Known limitations*). The tooling to produce them:

```bash
# list what you physically have, one per line: "brand model"
printf 'Puma Softride Enzo\nCampus North Plus\n' > data/my_shoes.txt

# resolve to catalogue ids BEFORE shooting -- anything unresolvable leaves the
# shoot list rather than poisoning the test set
python scripts/shotlist.py --have data/my_shoes.txt --items data/items.jsonl
```

That writes `data/shotlist.csv` (candidate SKUs with their colourway variants, so
you can pick the right one) and `data/shot_plan.md` (which conditions to induce
per shoe, and why half the plan is deliberately single-condition).

Then label them with the built-in tool rather than by hand in a spreadsheet:

```bash
vpm label --photos data/testset/photos --manifest data/testset/manifest.csv
```

It opens a local page at `127.0.0.1:8765` that runs the matcher on each photo and
offers its top candidates with thumbnails, so the common case is **confirming a
suggestion rather than typing an id**. Conditions are checkboxes drawn from the
one vocabulary in `src/vpm/eval/manifest.py`, and every row is validated before it
reaches the CSV. Keyboard-driven — `q`–`y` pick a match, `1`–`9` toggle
conditions, `Enter` saves and advances.

Why it exists: the two mistakes that are expensive here are assigning the wrong
catalogue id (which silently turns an in-catalogue photo into an out-of-catalogue
one — the exact bug that cost this project two evaluation runs) and drifting
condition vocabulary. Both are unrecoverable once 130 photos are shot, and both
are structurally impossible in this tool. Stdlib only — no web stack to install.

### Everything at once

```bash
./scripts/run_all.sh          # clean checkout -> reports/eval/report.md
pytest -q                     # 29 invariant tests, incl. the contamination guard
```

### Reproducing the backbone bake-off

```bash
python scripts/bakeoff.py --n-items 1200 --n-heldout 250 \
       --backbones siglip2-base dinov2-small-reg --whiten 0 256
```

---

## What each piece does

```
src/vpm/
  scrape/      sitemap harvesting, Myntra PDP parsing, resumable image download
  catalogue/   white-background trim, style clustering over the colourway graph
  embed/       backbone registry (SigLIP2 / DINOv2 ±registers / CLIP), saliency crop
  index/       exact flat index with view capping, PCA-whitening
  match/       query pipeline, open-set confidence features, conformal refusal
  eval/        Part B manifest schema, corruption operators, statistics, harness
```

**Design notes that are arguments, not plumbing** — each is defended in `REPORT.md`:

| Choice | Why |
|---|---|
| **SigLIP 2, not DINOv2** | Measured 0.672 vs 0.306 R@1 on this catalogue. My plan said DINOv2; the measurement overruled it. |
| **PCA-whitening** | Widens the same-item/different-item gap **15×**. Barely moves R@1. Separability and ranking are different axes. |
| **View capping at k=4** | Max-over-views inflates scores for items with more views (range here is 1–12), corrupting the refusal threshold. |
| **Style clusters** | 36% of naively held-out items kept a colourway twin in the index, making "absent" queries anything but. |
| **Conformal refusal** | The false-reject rate becomes a design parameter (α) rather than a tuned threshold. |
| **αQE / DBA default off** | Both inflate false accepts or destroy colourway distinctions. Built, measured, left off. |
| **Distractor bank** | Confidence is a likelihood ratio -- how much better the catalogue explains the photo than a generic pile of shoes does -- not an absolute similarity. |
| **Geometric verification as a feature** | One-sided evidence: measured 91 inliers on a self-match, 2 on a cross-match. Useless for ranking, useful for calibration. |
| **Saliency localiser off by default** | Built as the main answer to the domain gap, then measured: **identical** accuracy (hard R@1 0.556 either way) for 47% of the latency budget. `--localise` restores it. |
| **Stdlib server, not Streamlit** | ~30 transitive dependencies is a bad trade in a project whose gate is "runs from a clean checkout". |

---

## Data and provenance

Scraping is rate-limited (~10 req/s), identifies itself with a normal browser
user-agent, and records the source URL and fetch timestamp on every item.
`robots.txt` was checked before selecting the source: Myntra's is `Allow: /`
with disallows that do not cover product pages. Nike (`Disallow: */p/`), Adidas,
ASICS and New Balance were checked and rejected — the first by its rules, the
rest by anti-bot 403s.

`data/` and large artefacts are gitignored. The catalogue is reproducible from
step 1; nothing in the repo depends on a snapshot you cannot rebuild.

## Deployment

`scripts/package_deploy.py` emits a 27.5 MB bundle (index artefacts plus one
200px thumbnail per item — the 30 GB in the working tree is full-resolution
multi-view imagery kept for experiments). `docker build -t vpm-ui .` produces a
verified image: 825 MB RSS, 115 ms per lookup on two CPU threads.

See **[deploy/README.md](deploy/README.md)** for Cloud Run and Oracle Always Free,
and for the 2026 free-tier situation — HuggingFace Spaces withdrew free compute
Spaces, and Fly.io and Koyeb withdrew free tiers, so several obvious answers no
longer apply.

## Known limitations

Stated here rather than left to be discovered — the full accounting is in
`REPORT.md`:

1. **The bake-off protocol is pessimistic.** Holding out view 0 makes every query
   cross-view, which is harder than a real photo shot roughly side-on.
2. **The calibrator is fitted on synthetic corruptions** and would be deployed on
   real photos — a domain shift in the calibrator itself. The harness measures
   it (reliability diagram + ECE) rather than assuming it away.
3. **Catalogue snapshot drift.** A shoe bought two years ago may be delisted while
   a near-identical successor is listed. The manifest carries `sku_confidence`
   for exactly this, and metrics are reported with and without uncertain rows.
