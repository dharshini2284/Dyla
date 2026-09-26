# Visual Product Matcher — write-up

## Summary

A footwear matcher over a catalogue scraped from Myntra. Photograph a shoe, get
ranked candidates back with a calibrated confidence, and an explicit refusal when
the shoe is not in the catalogue.

The interesting content of this report is not the accuracy number. It is four
places where the obvious approach was tried, measured, and rejected — including
two where the thing I rejected was **my own plan**, and one where I rejected the
recommendation of the AI review I had commissioned.

| | |
|---|---|
| Footwear pool harvested | **345,331 SKUs** |
| Catalogue scraped (multi-view) | **36,506 items / 181,446 images / 30 GB** |
| Backbone | SigLIP 2 ViT-B/16 @256, PCA-whitened to 256d |
| Single-lookup latency | 35.7 ms (MPS) / 80.1 ms (CPU, 4 threads) — **search is ~1% of it** |

---

## 1. Catalogue: why Myntra

The binding constraint on this assignment is not the matcher. It is that Part B
requires photographs of items **genuinely in the catalogue**, which forces the
catalogue to overlap with shoes I can physically put in front of a camera.

I probed five sources before committing:

| Source | Result |
|---|---|
| Nike | `robots.txt` says "just crawl it" but `Disallow: */p/`; product-feed API 404s. **Rejected on its own rules.** |
| Adidas / ASICS / New Balance | 403 (Akamai). **Rejected.** |
| Puma India | Works — `__NEXT_DATA__`, Cloudinary images. Viable but **single-brand**. |
| Decathlon India | 9,148 products *across all categories*; footwear a fraction. **Too small.** |
| **Myntra** | `Allow: /`, product pages not disallowed. 277 gzipped sitemaps × 30,000 URLs ≈ **8.3M products**. **Selected.** |

Myntra wins on a property the others lack: **category and brand are in the URL
path** (`/Casual-Shoes/Big+Fox/<slug>/<id>/buy`), so the catalogue can be
filtered to footwear from the sitemaps alone, with zero page fetches. Sampling
120,000 URLs gave footwear at **4.17%** of the corpus → ≈346k footwear SKUs. The
full harvest found **345,331**, which is a nice check on the estimate.

It is also multi-brand, which matters because household shoes are not: Nike 1,483,
Adidas 2,461, Puma 3,853, Campus 1,910, Skechers 2,252, New Balance 901,
Reebok 2,222, Bata 2,065, Crocs 512, Converse 459. A single-brand catalogue would
have failed Part B outright.

### The metadata that turned out to matter most

`window.__myx` → `pdpData` yields everything in one fetch, including
`colours` — **the colourway-variant graph**. That is the single most valuable
field on the page, for a reason that only became clear once it broke something
(§4.2).

Measured on the catalogue: **18,810 items collapse to 12,380 style clusters**
(1.52 SKUs per style), and views per item range **1 to 12** (mean 6.08).

---

## 2. Retrieval approach

Views, not items, are the index rows — a query is one viewpoint, so collapsing
an item's views into a centroid smears the sole shot into the side shot and
throws away which view matched. Items are scored by **max over views**, then
whitened, then searched exactly.

Exact search is not a compromise here. Measured: brute-force cosine over the
catalogue is **under 1 ms**, roughly 1% of a lookup. Reaching for FAISS at this
scale optimises the wrong term by two orders of magnitude. An ANN index is
introduced only for the 100k tier, where it is still not the bottleneck.

---

## 3. The obvious approach, tried and rejected: DINOv2

My plan committed to **DINOv2** as the backbone, on solid-looking reasoning: this
is *instance*-level retrieval — find **this** shoe, not "a shoe" — and DINOv2's
self-supervised features are known to dominate instance retrieval, where CLIP
collapses visually distinct items into a shared semantic bucket.

That reasoning is correct **about landmarks**. The benchmarks DINOv2 wins
(ROxford, RParis) are rigid, textured 3D scenes under large viewpoint change.
Sneaker identity is carried by brand marks and colourway, which is a different
problem, and FORB (NeurIPS 2023) and ILIAS (CVPR 2025) both put CLIP/SigLIP
*ahead* on product retrieval.

So I ran a bake-off on my own catalogue rather than arguing from literature.
Protocol: index views 1..N, query with held-out view 0, hold whole style clusters
out of the index as genuine open-set negatives. Zero test-set budget spent.

| backbone | dim | clean R@1 | clean R@5 |
|---|---|---|---|
| **SigLIP 2 ViT-B/16 @256** | 1536 | **0.672** | **0.878** |
| DINOv2 ViT-S/14 +reg @224 | 768 | 0.306 | 0.481 |
| DINOv2 ViT-B/14 @224 | 1536 | 0.233 | 0.393 |

SigLIP 2 wins by **2.2×**. My plan was wrong, and it was wrong for a reason worth
naming: I generalised a benchmark result across a domain boundary it does not
cross. DINOv2 ViT-B also scoring *below* ViT-S is a second flag that the
landmark-tuned features are simply mismatched to this data.

**DINOv2 was not discarded.** It is retained as the *localiser* (§4.3), because
the thing it is genuinely best at — dense spatial structure — is what
localisation needs.

---

## 4. Five bugs I found in my own evaluation

The brief says it would rather read an honest account of a system at 70% than a
claim of 95% with no error analysis. These are the three places my own harness
was lying to me, found before any headline number was written down.

### 4.1 The corruption pipeline destroyed the signal outright

First hard-condition run: **R@1 = 0.001**. Not a weak result — a broken one.

Cause: I was corrupting **224px thumbnails**. Applying an 11-pixel motion-blur
kernel and then a 0.45× shrink to an already-downsampled image leaves nothing.
Real adverse conditions are *optical* and happen before the sensor downsamples.

Fix: corrupt at native resolution (1080px), then downsample. This is not a
tuning detail — a synthetic corruption applied at the wrong point in the pipeline
produces a benchmark that measures resampling, not robustness.

### 4.2 My open-set hold-out was not open-set

AUROC came back at **0.455 — worse than chance**, with held-out items scoring
*higher* than in-catalogue ones. A metric below chance is a bug, not a finding.

Cause: I held out individual items. **36% of them still had a colourway twin
sitting in the index.** The "absent" query had a near-identical match available,
so it was never absent.

Fix: build style clusters from the `colours` graph via union-find and hold out
**whole clusters**. This is exactly why that metadata field matters, and I would
not have known to look for it without the sub-chance AUROC.

### 4.3 The saliency crop was defeated by artefact tokens

The patch-token-norm localiser returned the full frame on every image. Cause:
DINOv2 without registers emits scattered high-norm artefact tokens in background
regions, so the bounding box of all above-threshold patches spans everything.

Measured on a real catalogue image:

| backbone | components | bbox rows |
|---|---|---|
| `dinov2-small` | **4** (9 stray patches) | 1–14 — inflated |
| `dinov2-small-reg` | **1** | 3–12 — correct |

Two fixes, both kept: use the register variant, and take the bounding box of the
**largest connected component** rather than of all above-threshold patches.

### 4.4 The eval path had never been run, and it was broken two ways

The gate is *"does it run from a clean checkout"*. I had built the entire
evaluation harness without once running `vpm eval` end to end. Doing so surfaced
two bugs that would otherwise have produced a report full of confident, wrong
numbers.

**The loud one.** The index was PCA-whitened to 256 dims; queries were encoded at
1536 and never passed through the whitener — `matmul: size 1536 is different from
256`. Fixed by carrying the whitener through the matcher, and by a constructor
check that refuses a dimension mismatch rather than discovering it at query time.

**The silent one, which was far worse.** The first successful run reported clean
Recall@1 of **0.067** against a bake-off baseline of 0.672 — while simultaneously
reporting the median rank of the true item as **1**. Both cannot be true, and
that contradiction is what made it findable:

```
items in index: 3000
distinct test items: 40
test items ACTUALLY IN THE INDEX: 6 (15%)
```

The test set was sampled from all 36,506 scraped items while the index held the
first 3,000. **34 of 40 "in-catalogue" items were never in the catalogue**, so
most "in-catalogue" photos were silently out-of-catalogue queries — accuracy
understated, and refusal AUROC collapsing to 0.502 because both classes were
really the same class.

The fix that matters is not the missing parameter. It is `check_items_indexed`,
which now **fails the run** if any photo labelled in-catalogue has an item absent
from the index. This is exactly the failure mode the brief warns about for Part B
— *"items that are genuinely in your catalogue"* — and a hundred hand-shot
photographs are far more expensive to get wrong than a synthetic set.

**The generalisable lesson:** a merely *bad* metric is easy to rationalise. Both
of these were caught because they were *impossible* — a 10× gap against a known
baseline, and a rank-1 truth that somehow was not the top-1 answer. It is worth
building enough redundant cross-checks that errors surface as contradictions
rather than as disappointing numbers.

### 4.5 The colourway graph is incomplete, and a single false accept exposed it

Found by *running the thing*, not by reading the code. Querying three
out-of-catalogue photos as a demo, one was accepted when it should have been
refused. The retrieved item was product `11168232` and the query was
`11168228` — adjacent ids, which is not what an unrelated shoe looks like.

```
11168228: Sparx Men Charcoal  Mesh Running Shoes   colour_variants = []
11168232: Sparx Men Navy Blue Mesh Running Shoes   colour_variants = [11904596]
```

Plainly the same model in two colours, and the catalogue's own `colours` field
does not link them. Since style clusters were built *entirely* from that field,
the two landed in separate clusters — so an item held out as "out of catalogue"
had its own colourway twin sitting in the index.

Measured across the full catalogue rather than assumed from one case:

```
items                                    36,506
clusters from the colours graph alone    23,450
model names split across clusters         3,457
links the colours graph misses            3,563  (9.8% of items)
```

The fix adds a second edge source: a **model key** derived from the product name
with colour words stripped, so "Sparx Men *Charcoal* Mesh Running Shoes" and
"Sparx Men *Navy Blue* Mesh Running Shoes" collapse to one key. Clusters went
**23,450 → 12,084** (merging is transitive, so the effect is larger than the raw
link count), and the exclusion set for a 20-item out-of-catalogue split grew from
39 product ids to 70.

**Two consequences, and the second is the one that matters.** Style-level
accuracy was understated. But worse, the *open-set evaluation was corrupted* in
the same way §4.2 was: "absent" items were not absent. Every refusal number
measured before this fix is therefore **pessimistic** — some false accepts were
the system correctly finding a near-identical shoe that the hold-out failed to
exclude.

This is the second time the same class of bug has bitten this project (§4.2 was
the first), which is itself the lesson: *near-duplicate leakage into a held-out
set is the default failure mode of open-set evaluation over a product catalogue*,
and a vendor's own variant metadata is not sufficient to prevent it.

---

## 5. The unexpected result: separability and ranking are different axes

Chasing the sub-chance AUROC produced the most useful finding in the project.

I measured same-item cross-view similarity against best-different-item
similarity, over 250 items:

| | cosine |
|---|---|
| Same item, different view | **0.9351** |
| Different item, best match | **0.9272** |
| gap | **0.0079** |

Identity is worth **under 1%** of the similarity. Everything else is viewpoint
and "this is a product photo of a shoe". The own-view-beats-other-item rate is
66%, which exactly predicts the 0.672 R@1 — the two numbers are the same fact.

This is the contrastive-embedding cone effect, and the textbook fix is
whitening. What was *not* expected is how unevenly it pays:

| | same | best other | **gap** | R@1 |
|---|---|---|---|---|
| raw | 0.9316 | 0.9263 | +0.0052 | 0.626 |
| whitened d=128 | 0.5965 | 0.5480 | +0.0485 | 0.639 |
| whitened d=256 | 0.5102 | 0.4360 | +0.0742 | 0.665 |
| whitened d=512 | 0.4188 | 0.3381 | **+0.0807** | 0.671 |

Whitening widens the separation gap **15×** while moving R@1 by **4.5 pp**.

If you evaluate a retrieval system on Recall@1 alone, whitening looks like a
rounding error and you might drop it. But the refusal problem lives entirely on
the axis whitening repairs: the decision "is this item in the catalogue at all"
is a question about the *margin*, not the *ranking*. **Ranking quality and
score-margin quality are separate properties of an embedding**, and the hardest
part of this brief depends on the one that R@1 does not see.

Whitening is fitted on catalogue embeddings only and never on queries.

---

## 6. Confidence and refusal

### Why raw cosine cannot work

Not "is imprecise" — *cannot*. A dark blurry photo of an in-catalogue shoe scores
low; a crisp photo of an absent shoe that resembles a catalogue one scores high.
Both error types move the **same direction** as a threshold slides, so no
threshold on top-1 similarity can fix both. Given §5, where identity is worth
0.008 of cosine, this is not a marginal concern.

### The scorer

Eighteen features → logistic regression → isotonic calibration. Grouped in four
blocks, each attacking a specific failure of raw cosine:

- **relative evidence** — margin to the next *distinct item*, Lowe's ratio at
  catalogue level, top-1 minus the mean of top-10 and top-100 (two bandwidths:
  one asks "is there a rival SKU", the other "is this item distinctive at all"),
  z-score against the query's own full similarity distribution
- **distractor normalisation** — top-1 minus the best match in a held-out
  distractor database. This is the block that breaks the tie raw cosine cannot: a
  dark in-catalogue photo has low top-1 *and* low distractor similarity; an
  absent shoe has high top-1 *and* high distractor similarity
- **self-consistency** — crop agreement across TTA crops, view agreement
- **query quality** — blur and exposure statistics

Logistic regression over a boosted tree deliberately: the coefficients are
printable, and "the distractor-normalised margin outweighs raw top-1 similarity"
is a claim a reader can check. A validation fit recovers exactly that ordering.

**The query-quality block is a stated tradeoff, not a free win.** It lets the
model learn "bad photo → refuse", which mechanically *raises* the false-reject
rate on the adversarial set being graded. The harness therefore fits the
calibrator both with and without that block and reports both risk–coverage
curves.

### Conformal refusal: FRR as a design parameter

Rather than tuning a threshold, the calibrated score feeds split conformal
prediction, which returns a **set** with a distribution-free coverage guarantee.
**The empty set is the refusal.** Verified:

| α | empirical coverage | nominal | mean set size | FRR | refusal on true negatives |
|---|---|---|---|---|---|
| 0.05 | 0.969 | 0.950 | 4.63 | 0.031 | 0.843 |
| 0.10 | 0.919 | 0.900 | 3.04 | 0.081 | 0.920 |
| 0.20 | 0.818 | 0.800 | 0.82 | 0.182 | 0.978 |

Coverage tracks nominal within 2 pp at every level, and **FRR ≈ α by
construction**. The brief asks for false-accept and false-reject rates; this
makes one of them a dial rather than an artefact. Set size is a free difficulty
readout: singleton means confident, eight means "one of these colourways", empty
means not in the catalogue.

**The guarantee's precondition is exchangeability between calibration and test,
and ours is violated by design** — the calibrator is fitted on synthetic
corruptions and deployed on real photos. That is a domain shift *in the
calibrator itself*. The harness measures it (reliability diagram + ECE) instead
of assuming it away, and the size of the coverage gap is the measurement of the
shift.

---

## 7. Latency

Measured with real weights at batch size 1, median of 30 runs:

| | MPS | CPU (4 threads) |
|---|---|---|
| DINOv2 ViT-B/14 @224 | 35.7 ms | 80.1 ms |
| DINOv2 ViT-S/14 @224 | 11.2 ms | 21.0 ms |
| exact search, full catalogue | <1 ms | <1 ms |

Measured end to end by `vpm bench` (SigLIP 2, 8,994 index rows / 3,000 items):

| stage | with localisation (3 crops) | **default, no localisation (2 crops)** |
|---|---|---|
| preprocess + localise | 28.42 ms | **0.57 ms** |
| backbone forward | 75.54 ms | **53.42 ms** |
| exact search + aggregation | 0.50 ms | **0.38 ms** |
| **total** | 104.44 ms | **54.35 ms** |

### The saliency localiser was measured and switched off

I built the patch-norm localiser as the main answer to the domain gap, debugged
DINOv2's artefact-token problem to make it work (§4.3), and added
largest-connected-component filtering on top. Then I rendered what it was
actually doing and found it selecting the **top 35% of a cluttered frame — pure
background — while the shoe sat in the middle**.

The obvious inference was that this explained the project's largest single
failure, cluttered background at 0/23 and −58 pp. So I ran the ablation:

| | localise ON | localise OFF |
|---|---|---|
| hard R@1 | 0.556 [0.467, 0.644] | **0.556** [0.456, 0.644] |
| hard R@5 | 0.656 | **0.656** |
| median latency | 112 ms | **60 ms** |

**The inference was wrong, and the feature is worthless.** Cluttered background
stays at 0/23 either way. Across all eight conditions the net effect is zero —
`off_angle` is better with it (+0.067), `motion_blur` better without (+0.048),
both one photo — and every other condition is bit-identical.

So localisation costs **47% of the latency budget and buys nothing measurable**.
It is now **off by default**, recoverable with `--localise`.

Two things worth separating here. First, this is a negative result about a
component I had already invested in and had good reasons to believe in — the
domain gap is real, and localising the object is the textbook answer to it.
Second, it was only findable by *looking at the output*: the numbers alone said
"cluttered background is bad", and it took the overlay to show the localiser was
choosing background, which prompted the ablation that showed it did not matter
either way.

I am **not** removing it, for a stated reason: the synthetic clutter shrinks the
shoe into high-frequency noise, which is a harsher and less realistic input than
a real photo of a shoe on a carpet. On hand-shot photographs localisation may
well earn its cost. The flag stays, the default reflects what is measured today,
and the question is explicitly open.

**The backbone forward pass is the entire budget — search is half a percent of
it.** Anyone reaching for an ANN index at this scale is optimising the wrong term
by more than two orders of magnitude, and the way to get faster here is to make
fewer forward passes (drop a TTA crop, skip localisation on easy queries), not to
change the index. Every meaningful latency
decision is about how many forward passes to make (localisation, TTA), not about
index structure.

For the 100k / CPU-only / <100 ms target, ViT-B costs 80 ms on CPU and leaves no
headroom, so that tier uses ViT-S at 21 ms, leaving ~79 ms for everything else —
comfortably enough, since search is ~1 ms even at 100k.

### Where I overruled the AI review

I commissioned an architecture review, and it was right about SigLIP 2, the
view-count bias, and the αQE false-accept risk. It was **wrong** about one thing:
it reported that CPU is faster than MPS at batch size 1 and recommended splitting
device policy by call site.

It had measured *synthetic shape-accurate transformer stacks*, not real timm
graphs. Re-measured with real weights, MPS is **~2× faster** in both model sizes.
I kept MPS for serving. Its downstream conclusion for the CPU-only tier survived
its wrong premise, which is its own lesson: a right answer from a wrong
measurement is still worth re-deriving.

---

## 8. Part B: what is built and what is not

**Not in this repo: the 100 hand-shot photographs.** Part B requires photographs
of items the author physically owns, and I will not present synthetic images as
hand-shot ones.

What *is* built and tested:

- **Manifest schema and validator** — 13-condition vocabulary, `occludes_logo`
  (because a hand over the logo is categorically worse than the same area of
  midsole), and `sku_confidence ∈ {exact, style_match, uncertain}` for catalogue
  drift. The validator rejects unknown conditions, unknown item ids, duplicate
  photo ids, and conditions on clean controls — verified against a manifest with
  six seeded errors, all six caught.
- **A mechanical contamination guard** — `check_split_disjoint` fails if any
  *item* appears in both dev and test. Item-disjoint, not merely photo-disjoint:
  two photos of one shoe share an object, a rig and a session.
- **Nine corruption operators** at five severities, with low light coupled to
  motion blur because a dark scene forces a longer exposure. That coupling is
  deliberate — an orthogonal synthetic design would make the marginal-effects
  analysis look far easier than reality.
- **The full statistics stack**, validated against synthetic ground truth: it
  correctly identified a genuinely damaging condition (−27.8 pp, CI excludes 0),
  an inert one (−2.5 pp, CI includes 0), and showed a *confounded* condition only
  partially disentangled (−17.8 pp) — which is the honest caveat about what this
  design can and cannot resolve.
- **A synthetic stand-in test set** exercising the identical code path, so every
  table is reproducible from a clean checkout today.

### The methodology, and why

- **Bootstrap resamples items, not photos.** Photos of one item are correlated;
  i.i.d. photo resampling understates variance.
- **Per-condition cells are not rankable at this n.** Wilson on 8/12 gives
  [0.391, 0.862]. The report prints the interval next to every cell and states
  that the *ordering* is unresolvable, rather than ranking noise. The condition
  ranking should be read from the marginal-effects table and the synthetic
  dose–response sweep, which have n in the thousands.
- **Conditions co-occur causally**, so per-condition accuracy is confounded.
  Marginal effects are estimated by penalised logistic regression on the
  condition indicator matrix, bootstrapped by item, and labelled **exploratory**.
- **The clean control set is paired by item and tested with exact McNemar.**
  Pairing removes item difficulty, which is a real confounder because the
  photographer's choice of which shoes to shoot under which conditions is not
  random. The paired difference *is* "the gap between the two halves".
- **Accuracy is reported at SKU and style level.** Many SKUs differ only in a
  colourway panel; at SKU level some photos pose an unanswerable question.
  Reporting only SKU accuracy takes a penalty for an impossible task, and
  reporting only style accuracy hides the hard part.
- **Three-way outcome table**, including the **wrong-accept rate** — the error a
  user actually feels, which a bare FAR/FRR pair hides.

---

---

## 9. Results

Produced by `vpm eval` (`reports/eval/report.md`). **These are from the synthetic
stand-in set, not hand-shot photographs**, and must not be read as a Part B
result. What they demonstrate is that the measurement machinery works and that it
already surfaces a real, specific failure.

Catalogue: 3,000 items / 8,994 views, SigLIP 2 whitened to 256d.
Test split: 90 hard, 30 clean, 20 out-of-catalogue.

| set | R@1 SKU | R@1 style | R@5 SKU | median latency |
|---|---|---|---|---|
| clean | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 159 ms |
| hard | **0.556** [0.467, 0.644] | 0.567 | 0.656 | 160 ms |

**The clean 1.000 is an artefact and I am not claiming it as a result.** In the
synthetic set the "clean control" *is* the indexed image, so the match is exact
by construction. Real clean photographs would be different images of the same
object and would score well below 1.0. The figure is useful only as a sanity
check that indexing and retrieval are wired correctly — which, given §4.4, is not
a trivial thing to confirm.

### The gap between the two halves

Paired by item, exact McNemar: **40 clean-right/hard-wrong, 0 in the other
direction. A 44.4 pp drop, p = 1.8 × 10⁻¹².** The conditions cost roughly half
the system's accuracy, and the direction is completely one-sided.

### Which conditions actually hurt

Per-condition accuracy is confounded (conditions co-occur by construction, as
they do in reality), so the marginal-effects model is what the ranking should be
read from:

| condition | n | R@1 | marginal effect | 95% CI on log-odds |
|---|---|---|---|---|
| **cluttered_background** | 23 | **0.000** | **−58.1 pp** | [−3.34, −2.45] |
| low_light | 17 | 0.412 | −22.2 pp | [−2.05, −0.26] |
| partial_occlusion | 10 | 0.500 | −5.6 pp | [−1.46, +0.63] |
| small_in_frame | 12 | 0.333 | −4.2 pp | [−0.84, +0.37] |
| defocus | 22 | 0.682 | +2.3 pp | [−0.42, +0.70] |
| motion_blur | 21 | 0.667 | +4.9 pp | [−0.46, +1.10] |
| specular_reflection | 14 | 0.786 | — | — |

Only **two** conditions have intervals excluding zero. Everything else is
indistinguishable at this sample size, and the report says so rather than ranking
noise.

**Cluttered background is catastrophic — 0 correct out of 23.** That is the
single most actionable finding, and it points straight at §4.3: the saliency
localiser is what is supposed to handle clutter, and it evidently does not. The
conditions I expected to dominate — motion blur, defocus — are not
distinguishable from zero effect. I would not have guessed that ordering.

Dose response: each additional adverse condition multiplies the odds of a correct
match by **0.366**. Accuracy falls 0.709 (one condition) → 0.308 (two).

### Error taxonomy

Every failure is auto-classified rather than described in prose
(`src/vpm/eval/taxonomy.py`), using the colourway graph plus brand and article
type: `colourway_confusion` (right style cluster, wrong SKU),
`same_brand_confusion`, `silhouette_confusion` (same article type, different
brand), `catastrophic`, and `sku_absent` (the true item was never in the index —
not a retrieval failure at all). The per-class table is emitted by `vpm eval`
into `reports/eval/report.md`.

The distinction that carries the most information is colourway confusion: at SKU
level it is a miss, but it is a categorically different failure from returning an
unrelated shoe, and one number for both hides which problem you actually have.

**And the result contradicts what I predicted.** Throughout the design I treated
colourway confusion as the error mode that would dominate — it is why the
catalogue deliberately imports near-identical variants, and why accuracy is
reported at style level as well as SKU level. Measured:

| class | n | share of errors |
|---|---|---|
| silhouette_confusion (same article type, different brand) | 18 | **0.450** |
| catastrophic (unrelated article type) | 13 | 0.325 |
| same_brand_confusion | 8 | 0.200 |
| **colourway_confusion** | **1** | **0.025** |

Colourway confusion is **2.5% of errors**, not the bulk of them. The system is
not failing to tell near-identical SKUs apart; it is failing to tell a Puma
sneaker from an unrelated brand's sneaker, and a third of the time it returns
something in a different category entirely. That is consistent with §5 — with
identity worth 0.008 of cosine, the model is matching *silhouette and product-shot
style*, not instance. It also means the style-level metric buys almost nothing
here (0.567 vs 0.556), which I had expected to be a much larger gap.

I record this because the prediction was wrong in a way that would have changed
where I spent effort: hard-negative mining over the colourway graph, which the
plan treats as central, is aimed at 2.5% of the errors.

### Refusal

Three configurations, in the order they were measured. The third is the one that
counts — the first two were taken before §4.5 was found, so their hold-out leaked
colourway twins.

| | no distractor bank | + distractor bank | **+ corrected clusters** |
|---|---|---|---|
| AUROC, in-catalogue vs out | 0.726 | 0.850 | **0.827** |
| Calibrated vs raw-cosine AUROC | 0.924 / 0.890 | 0.935 / 0.873 | **0.924 / 0.876** |
| False accept rate | 0.550 | 0.300 | **0.250** |
| False reject rate | 0.258 | 0.258 | **0.292** (nominal α = 0.10) |
| **Wrong-accept rate** | 0.100 | 0.083 | **0.075** |
| AURC / E-AURC | 0.147 / 0.027 | 0.143 / 0.023 | **0.150 / 0.029** |

**What is solid, and what is not.** The distractor bank is a genuine, large effect
— column 1 to column 2 halves the false-accept rate, and the fitted coefficients
confirm the mechanism (`s1_minus_distractor_max` is consistently among the top
features, and the calibrated score's margin over raw cosine roughly doubles).

**The column 2 → column 3 movement is not a result and I am not claiming it as
one.** With 20 negatives, FAR 0.300 → 0.250 is *one photo*. AUROC moved the other
way (0.850 → 0.827) and FRR rose. I predicted in §4.5 that the leaky hold-out
made the earlier numbers pessimistic; the corrected run is consistent with that
but **cannot confirm it at this sample size**, and the honest summary is that the
third column is measured on a cleaner and genuinely harder hold-out — 70 excluded
product ids instead of 39 — so the configurations are not directly comparable
anyway.

This is precisely why the brief asks for 20 out-of-catalogue photographs and why
that number is uncomfortably small: a 5-percentage-point move in FAR is a single
image, and no amount of careful statistics rescues that.

**The distractor bank is the single largest improvement in the system**, and it
is the one component whose value I argued for on theory before measuring. Adding
4,000 held-out shoes — absent at *style* level, so no colourway twin leaks back —
cut the false-accept rate nearly in half and moved AUROC 0.726 → 0.850. The
fitted coefficients confirm the mechanism rather than just the outcome:
`distractor_max` is the second-strongest feature in the model and
`s1_minus_distractor_max` the fourth, and the margin of the calibrated score over
raw cosine roughly doubled (+0.034 → +0.062).

That is the likelihood-ratio argument from §6 doing exactly what it was supposed
to: asking *how much better does the catalogue explain this photo than a generic
pile of shoes does*, instead of asking whether a similarity clears a threshold.

**This is the weakest part of the system and I am not going to dress it up.** A
FAR of 0.55 means the refuser accepts more than half of the out-of-catalogue
shoes. With only 20 negatives the interval on that is very wide, but the point
estimate is bad.

**The conformal coverage guarantee did not hold**: FRR came out at 0.258 against
a nominal 0.10. This is the exchangeability violation predicted in §6 —
calibration is fitted on synthetically corrupted catalogue views, and the test
queries are a different distribution. The guarantee is only as good as its
precondition, and here the precondition is false. That is the honest reading, and
it is why the design measures coverage rather than asserting it.

The diagnosis is available rather than speculative: out-of-catalogue items are
*other shoes*, and §5 showed identity is worth 0.008 of raw cosine. A different
shoe is not far away in this space. Whitening widened that gap 15×, which is why
the calibrated AUROC (0.924) beats raw cosine (0.890) — but the remaining margin
is still thin, and a 20-negative sample cannot resolve where the threshold
belongs. The fix is more negatives and a larger distractor database, not a
different threshold.

### Growing the catalogue without recomputing (`scripts/incremental_add.py`)

The extension asks to add a thousand items "without recomputing anything you
already had" and show accuracy on the originals is unchanged. Claiming that is
easy, so the script emits two certificates instead:

| | |
|---|---|
| items added | 1,000 |
| **embeddings recomputed** | **0** |
| **pre-existing embeddings bit-identical** | **True** (`np.array_equal` on raw float32, not `allclose`) |
| Recall@1 on the *original* items | 0.1667 → 0.1467 (**−2.0 pp**) |

`array_equal` rather than a tolerance is deliberate: a tolerance would hide
exactly the drift being tested for.

**The −2 pp is real and is not a bug.** A thousand new items are a thousand new
competitors for the same queries, so a small drop is the correct behaviour; what
must not happen is the *old vectors* changing, which certificate 1 covers. The
probe uses deliberately perturbed queries — a query taken verbatim from the index
self-retrieves at cosine 1.0, saturating the metric at 1.000 and certifying
nothing even if the insert had wrecked the neighbourhood structure. (My first
version of this test did exactly that and reported a meaningless 1.000 → 1.000.)

This works because two earlier choices, both made for other reasons, happen to
make insertion a concatenation: the whitener is **fitted once and frozen** (a
refit would rotate every existing vector and break certificate 1), and view
capping is per item, so it never reaches across items.

**The cost, stated rather than hidden:** a frozen whitener drifts from the
optimal transform as the catalogue grows. This buys O(1) insertion at the price
of a transform that slowly goes stale and eventually needs an offline refit — at
which point every embedding *does* have to be recomputed.

## 10. What does not work, and what I did not do

0. **Refusal is the weak point**: FAR 0.550 at a wrong-accept rate of 0.100, and
   conformal coverage missed nominal (FRR 0.258 vs α = 0.10). Diagnosed in §8,
   not hidden.
1. **Cross-view retrieval is the ceiling, and it is low.** At 0.672 R@1 on clean
   held-out views, a third of clean queries already fail. §5 explains why:
   identity is 0.8% of the signal. Whitening treats the symptom on the margin
   axis; it does not make the backbone view-invariant.
2. **The bake-off protocol is pessimistic and I have not corrected for it.**
   Holding out view 0 makes every query cross-view, whereas a real photo is
   usually roughly side-on like the hero shot. Real-photo accuracy should be
   *higher* than 0.672, but by an amount I have not measured and will not guess.
3. **αQE and DBA are built (`src/vpm/match/rerank.py`) and default to off.** αQE
   drags the query toward the catalogue manifold, inflating top-1 for
   out-of-catalogue queries — it raises exactly the false-accept rate the hardest
   extension is graded on, so any confidence feature must be computed on the
   pre-expansion query. DBA averages colourway clusters toward their centroid,
   destroying the distinction that matters most here. Two negative results, kept
   behind flags rather than deleted.
4. **Geometric verification is a confidence feature, not a re-ranker**, and the
   asymmetry is measured rather than asserted. On two real catalogue images,
   SIFT + RANSAC (affine, not homography — a shoe upper deforms and the planar
   assumption costs inliers on genuine matches) gives **91 inliers on a
   self-match and 2 on a cross-match**. High counts are near-proof; low counts
   mean "low texture", not "not a match". One-sided evidence is useless for
   ordering the low-texture majority and useful as one calibrator input.
5. **The synthetic→real calibration gap is unmeasured**, because measuring it
   requires the real photos. It is the largest known unknown in the design.
6. **Not attempted:** the multi-item extension, and the domain-adaptive
   projection head. Both were planned; both were cut to keep the error analysis
   deep rather than the feature list long, which is what the brief asks for.

## 11. Two things built that the brief did not ask for

Both exist because of something measured, not because a UI seemed nice.

**A labelling tool (`vpm label`).** The engineering bottleneck in this project is
not the matcher, it is getting 130 photographs labelled without introducing the
errors that quietly corrupt every downstream number. Two mistakes are expensive
and easy to make in a spreadsheet: assigning the wrong catalogue id — which
silently converts an in-catalogue photo into an out-of-catalogue one, the exact
bug that cost this project two evaluation runs (§4.4) — and drifting condition
vocabulary. The tool makes both structurally impossible: it runs the matcher on
each photo and offers ranked candidates with thumbnails so ids are *confirmed*
rather than typed, and conditions are checkboxes bound to the single vocabulary
in `eval/manifest.py`. Stdlib only; a reviewer should not need a web stack to
label photographs.

**A visual error report (`reports/eval/errors.html`).** `report.md` already has
every number. This shows each failure as query / returned / correct, side by
side, with its auto-assigned error class and induced conditions. It is the one
thing a table cannot do — a reader can see that a `silhouette_confusion` really
is two shoes of the same shape, rather than taking the classifier's word for it.
Self-contained, thumbnails embedded.

**An interactive matcher (`vpm ui`).** Drop a photo, see the ranked matches, the
calibrated confidence, the conformal set, an explicit refusal badge *with its
reason*, a per-stage latency breakdown and the saliency overlay.

I built this last and reluctantly — a demo page shows the part that already
works. It earned its place for an unplanned reason: **rendering the saliency
overlay is what exposed that the localiser was worthless** (§7). The numbers
alone said "cluttered background is bad"; it took seeing the box land on empty
background to prompt the ablation that removed 47% of the latency. That is an
argument for building the visualisation, not for building the demo.

Stdlib `http.server` throughout. Streamlit would have been faster to write and
would have added roughly thirty transitive dependencies to a project whose gate
is *"does it run from a clean checkout"* — a bad trade here.

**On reproducibility:** `requirements.lock` is regenerated from the working
environment, and it caught real drift — numpy moved 1.26 → 2.4.6 and torch
2.11 → 2.14 mid-project, which broke `ndarray.ptp()` in the overlay renderer.
A lock file is not bureaucracy on a project a reviewer is expected to re-run.

## 12. What I would do next, in order

1. Shoot the 130 + 30 photographs, lock the split, and measure the
   synthetic→real calibration gap. Everything else is downstream of this.
2. Resolution sweep at 224 / 256 / 384 / 512. Given that identity is 0.8% of the
   signal, I suspect this system is **resolution-limited, not model-limited** —
   a very different conclusion from "the model isn't good enough", and cheap to
   test.
3. The domain-adaptive projection head, trained contrastively on
   (clean, corrupted) pairs with hard negatives mined from the colourway graph.
   §5 says the clean/corrupted axis is where the variance is.
