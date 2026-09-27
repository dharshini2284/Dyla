# Real-world test set: publicly sourced user photos

184 real photographs of footwear. People took them with ordinary cameras and
phones and uploaded them to Wikimedia Commons. **None were taken by the author of
this repository, and none are catalogue or manufacturer images.** They are here
to test the matcher on the kind of photo a user would actually submit: worn on
feet, held in hands, on carpets, grass, gravel, shoeboxes, in low light and at
odd angles.

| | photos | test | dev |
|---|---|---|---|
| `catalogue`: footwear that is in the catalogue | **115** | 100 | 15 |
|   of which `exact` (model and colourway listed) | 71 | 67 | 4 |
|   of which `style_match` (model listed, colourway not) | 44 | 33 | 11 |
| `out_of_catalogue`: footwear that is not | **69** | 54 | 15 |
| **total** | **184** | 154 | 30 |

The catalogue photos cover **42 SKUs across 20 catalogue models** (Converse,
Nike, adidas and Birkenstock). The out-of-catalogue photos cover **22 footwear
types**. The dev and test splits are **item-disjoint**, checked by
`vpm.eval.manifest.check_split_disjoint`. The locked test split holds exactly 100
catalogue photos, as the brief requires.

## Files

| file | what it is |
|---|---|
| `images/catalogue/<item_id>_<photo_id>.*` | catalogue photos (gitignored; see *Rebuilding*) |
| `images/out_of_catalogue/<type>_<photo_id>.*` | out-of-catalogue photos (gitignored) |
| `labels.csv` | one row per image: `image_path, label, catalogue_status, catalogue_item, item_id, sku_confidence, split` |
| `metadata.csv` | everything in `labels.csv`, plus source URL, author, licence, date, the exact download URL, SHA-256, image size, induced-condition tags, and the labeller's notes |
| `manifest.csv` | the same set in the `vpm eval` schema (`src/vpm/eval/manifest.py`); validates cleanly |
| `index_items.txt` | the 42 catalogue item IDs these photos are labelled with (see *Using it with `vpm eval`*) |
| `screening_log.csv` | **all 1,492 candidates looked at**, each with its keep/reject decision and the reason |
| `summary.json` | counts, generated alongside the CSVs |
| `ATTRIBUTION.md` | author and licence for every image |

## Label definitions

- **`catalogue_status = catalogue`**: the photo shows a shoe model that the
  Myntra catalogue (`data/items.jsonl`) lists. `item_id` is the single catalogue
  SKU it is scored against. `equivalent_ids` in `metadata.csv` lists other SKUs of
  the same model whose *listed* colour matches. That list comes from the
  catalogue's `base_colour` field, which is unreliable (see *Limitations*). Treat it
  as candidates, not as ground truth.
- **`sku_confidence = exact`**: the model and colourway visibly match the chosen
  SKU. Each exact group's `item_id` was pinned by comparing the photo side by side
  with that SKU's catalogue image. A matching `base_colour` was not enough.
- **`sku_confidence = style_match`**: the model is in the catalogue but this
  colourway or edition is not. Examples: a Jordan 1 *High* when only Mids are
  listed, a black/white Samba when only tonal and coloured-stripe Sambas are
  listed, a red/white Air Max 1 when only black and brown are listed. At SKU level
  these photos have no correct answer. In the manifest's own terms they are "the
  hardest refusal negatives", and metrics should be reported with and without them.
- **`catalogue_status = out_of_catalogue`** (`label = out_of_catalogue`): genuine
  footwear that is **not** in the catalogue. Two kinds, recorded in `model_key`:
  - *type absent*: flip-flops, rubber boots, ski boots, moon boots, figure and
    roller skates, pointe shoes, UGG boots, Crocs clogs, Birkenstock Arizona/Gizeh
    sandals. The catalogue is closed shoes and boots.
  - *brand or model absent*: Dr. Martens, Vans, Onitsuka Tiger, Reebok Freestyle
    Hi and Princess, Nike Air Max 97 / 720 / Air Ship, New Balance 480, Skechers
    Energy. Several of these are deliberately **hard** negatives: the brand is
    listed and the silhouette is close. Air Ship is the Jordan 1's predecessor, and
    the Air Max 97 sits beside listed Air Max models.

  Absence was checked by searching all 36,506 catalogue names for each model. That
  check is how the three Air Max 1 photos were caught. They had been screened as
  out-of-catalogue, but two "Air Max 1 Essential" SKUs exist, so they moved to
  `catalogue / style_match`. No out-of-catalogue photo is forced into a catalogue
  label.
- **`conditions`** (catalogue rows): the realistic difficulty visible in the
  photo, using the manifest's fixed vocabulary. Counts: `hand_or_foot_in_frame` 38,
  `cluttered_background` 21, `off_angle` 19, `low_light` 14, `worn_dirty` 11,
  `partial_occlusion` 8, `cropped_item` 4, `specular_reflection` 2, `motion_blur` 2,
  `small_in_frame` 1, `harsh_flash` 1. 31 photos carry no tag; they are real photos
  of a shoe in normal light. These tags are **observed**, not induced as in the
  shot plan, so they are unbalanced and correlated. Read per-condition numbers as
  descriptive only.

## How it was collected

1. **Source selection, with the rules checked first.**
   - *Myntra customer-review photos* would have been ideal, since they are the same
     SKUs photographed by buyers. **Rejected:** Myntra's Terms of Use forbid any
     "page-scrape, robot, spider or other automatic device… to access, acquire,
     copy or monitor any portion of the Platform or any Content". That covers
     user-generated content.
   - *Openverse* (a Creative Commons search engine that indexes Flickr).
     **Rejected:** its `robots.txt` disallows `/v1/images/` for every agent and
     names `anthropic-ai` specifically. One sample request was sent before that rule
     was read. It was not used, and nothing else was requested.
   - *Flickr API*: needs an account-bound key. Not used.
   - *Wikimedia Commons*. **Used.** Everything on it is freely licensed, and much
     of it is ordinary people photographing their own shoes. Its `robots.txt`
     disallows the `/w/` API for generic crawlers, but says "friendly, low-speed
     bots are welcome viewing article pages". So only the ordinary
     `/wiki/Category:…` and `/wiki/File:…` pages were read, plus images from
     `upload.wikimedia.org` (allowed). Pagination links, which go through `/w/`,
     were not followed.
   - Access pattern: one request per second, a descriptive User-Agent, backoff on
     429/503. No login, CAPTCHA, paywall or access control was involved or
     bypassed.
2. **Candidates.** File lists were taken from model-level Commons categories
   (e.g. *Chuck Taylor All-Stars (high-top)*, *Adidas Samba*, *Nike Air Force*,
   *Dr. Martens*, *Crocs*, *Ski boots*), chosen after checking which models the
   catalogue lists. Logos, stores, maps and non-photo formats were dropped by
   filename. **1,492 candidates** remained.
3. **Visual screening, every candidate by eye.** Each candidate was viewed and
   either kept or rejected with a reason, all recorded in `screening_log.csv`.
   1,248 were rejected. Approximate reasons, grouped by keyword:
   - about 520 retail, store, advert or display
   - about 190 shoe too small in frame, person-centric photo, or sports event
   - about 145 museum, archive or historic model
   - about 140 not footwear or model unidentifiable
   - 84 electronics "flip-flop" circuit diagrams (a category-name collision)
   - about 65 near-duplicates or the same photo session
   - about 60 customised or special editions
   - about 45 clean product or studio shots
4. **Labelling.** Each keeper got a model, cut (high/low), colourway and
   condition tags. It was then mapped to catalogue SKUs by name and colour, and
   **every `exact` row was re-checked side by side against its SKU's catalogue
   image**. That check changed a lot:
   - it caught SKUs tagged "White" that are maroon or blue-soled
   - it caught "Youth" SKUs chosen as primaries
   - it caught a photo whose title said black but which was navy
   - it downgraded 15 rows from `exact` to `style_match` where the colourway was
     only close

   Twenty-three screened keepers were dropped at this stage: cut not
   determinable, colour ambiguous, customised, or a cap per SKU reached (at most
   9 photos per SKU, to stop black and red Chucks dominating). Out-of-catalogue
   types were capped at 5 each for diversity, which excluded 36 more.
5. **Download and de-duplication.** For each kept file, the Commons page was
   read for author, licence and date, and a 1280px Commons rendition was fetched
   (the original if smaller). Duplicate check: perceptual (dHash, Hamming distance
   ≤ 5) plus exact SHA-256 across the final set. **0 duplicates found.** Same-object
   series had already been cut at screening, and those cuts are counted in the
   screening log, not here. One file failed to download and is excluded.
6. **Splits.** Whole catalogue items with the fewest photos go to dev until test
   holds exactly 100 catalogue photos. About 1 in 6 out-of-catalogue *types* go to
   dev. Items and types never straddle the two splits.

## Rebuilding

The images are gitignored; `metadata.csv` is not. From a clean clone:

```bash
python scripts/fetch_real_world.py        # ~184 requests at 1/s, ~65 MB
```

It downloads each file from the exact URL recorded in `metadata.csv`, contacts
only `upload.wikimedia.org`, and verifies each file against its recorded SHA-256.

## Using it with `vpm eval`

The harness refuses to score a catalogue photo whose item isn't indexed
(`check_items_indexed`, the guard from `REPORT.md` §4.4). **Only 10 of the 42
labelled SKUs are in the current 3,000-item index.** The rest, including every
Chuck Taylor, are in the 36,506-item catalogue with images already downloaded,
but they were never encoded. They must be added before evaluation:

```bash
# add the labelled items (plus the usual catalogue) to an index, then
vpm eval --index data/index_rw.npz --items data/items.jsonl \
         --testset data/real_world --out reports/eval_real_world
```

`index_items.txt` lists the IDs to include. This run has **not** been done yet,
so no real-world accuracy number exists at the time of writing.

## Limitations

- **Coverage is skewed by what the public photographs.** Converse (55 photos) and
  Nike (37) dominate the catalogue side. Indian mass-market brands that make up
  much of the catalogue (Campus, Sparx, Woodland, Bata, Red Tape) have essentially
  no freely licensed user photos, so they are not represented. On this set, the
  matcher is tested on famous silhouettes, which are probably easier than the
  catalogue average.
- **Short of 100 `exact` rows.** 115 photos are catalogue footwear, but only 71
  match both model and colourway. The other 44 are `style_match`. Reaching 100
  exact-SKU photos from openly licensed sources would need model categories that
  Commons does not have, or a source whose terms permit collection.
- **`exact` means model and colourway, not the literal SKU.** The shoe in a
  public photo can't be traced to a listing. It may be an older production run, a
  different size or a regional variant of the same colourway.
- **The catalogue's `base_colour` is noisy.** This is why primaries were pinned by
  eye, and why `equivalent_ids` should not be treated as verified.
- **The `conditions` tags are observed, not controlled,** and they are one
  labeller's judgement.
- **Not photos by the author.** This set does not replace the hand-shot set the
  brief describes. It is an independent, publicly sourced stand-in that is much
  closer to real user photos than the synthetic-corruption set.
