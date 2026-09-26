## Results — split `test`

140 photos: 90 hard, 30 clean, 20 out-of-catalogue

### Headline

| set | R@1 SKU | R@1 style | R@5 SKU | median rank of truth | median latency |
|---|---|---|---|---|---|
| clean | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1 | 60 ms |
| hard | 0.556 [0.456, 0.644] | 0.567 [0.467, 0.656] | 0.656 [0.578, 0.733] | 1 | 60 ms |

95% intervals are cluster bootstrap resampling **items**, not photos.

### The gap between the two halves (paired by item, McNemar exact)

- clean-right/hard-wrong: **40**, clean-wrong/hard-right: **0**
- accuracy drop: **+44.4 pp** over 90 pairs, p = 1.819e-12

### Error taxonomy

| class | n | share of all | share of errors |
|---|---|---|---|
| correct | 50 | 0.556 | — |
| colourway_confusion | 1 | 0.011 | 0.025 |
| same_brand_confusion | 9 | 0.100 | 0.225 |
| silhouette_confusion | 14 | 0.156 | 0.350 |
| catastrophic | 16 | 0.178 | 0.400 |

> `colourway_confusion` is the right model in the wrong colour. At SKU level it is a miss, but it is a categorically different failure from returning an unrelated shoe, and reporting one number for both hides which problem you actually have.

### Accuracy by failure condition

| condition | n | R@1 SKU | 95% Wilson | R@1 style |
|---|---|---|---|---|
| cluttered_background | 23 | 0.000 | [-0.000, 0.143] | 0.000 |
| small_in_frame | 12 | 0.333 | [0.138, 0.609] | 0.417 |
| low_light | 17 | 0.412 | [0.216, 0.640] | 0.412 |
| off_angle | 15 | 0.467 | [0.248, 0.699] | 0.467 |
| partial_occlusion | 10 | 0.500 | [0.237, 0.763] | 0.500 |
| defocus | 22 | 0.682 | [0.473, 0.836] | 0.682 |
| motion_blur | 21 | 0.714 | [0.500, 0.862] | 0.714 |
| specular_reflection | 14 | 0.786 | [0.524, 0.924] | 0.786 |

> Cells are small. At n=10 a 95% interval spans roughly ±25 pp, so the **ordering** of these conditions is not resolvable from this set. The marginal-effects table and the synthetic dose-response sweep are what the condition ranking should be read from.

### Marginal effect of each condition (exploratory)

| condition | n | log-odds | 95% CI | avg marginal effect |
|---|---|---|---|---|
| cluttered_background | 23 | -2.86 | [-3.26, -2.39] | -57.0 pp |
| low_light | 17 | -1.35 | [-2.16, -0.45] | -24.5 pp |
| partial_occlusion | 10 | -0.27 | [-1.40, +0.67] | -4.7 pp |
| small_in_frame | 12 | -0.21 | [-0.82, +0.39] | -3.7 pp |
| off_angle | 15 | +0.12 | [-0.57, +0.75] | +2.0 pp |
| defocus | 22 | +0.14 | [-0.41, +0.69] | +2.3 pp |
| specular_reflection | 14 | +0.31 | [-0.38, +1.22] | +5.3 pp |
| motion_blur | 21 | +0.55 | [-0.15, +1.33] | +9.4 pp |

> Conditions co-occur, so per-condition accuracy above is confounded. This fits correctness on the condition indicator matrix to estimate each condition's marginal contribution. Penalised and bootstrapped by item; **exploratory** at this sample size.

### Dose response

Each additional adverse condition multiplies the odds of a correct match by **0.366**.

| # conditions | n | accuracy | 95% Wilson |
|---|---|---|---|
| 1 | 55 | 0.709 | [0.579, 0.812] |
| 2 | 26 | 0.308 | [0.165, 0.500] |
| 3 | 9 | 0.333 | [0.121, 0.646] |

### Refusal

- AUROC in-catalogue vs out-of-catalogue: **0.837** (120 in, 20 out)

| outcome | count / rate |
|---|---|
| in-catalogue, accepted & correct | 78 |
| in-catalogue, accepted & **wrong** | 10 |
| in-catalogue, refused (FRR) | 32 (0.267) |
| out-of-catalogue, accepted (**FAR**) | 0.250 |
| **wrong-accept rate** | 0.083 |

> The wrong-accept rate is the error a user actually feels, and a bare FAR/FRR pair hides it.

- AURC **0.147**, excess over oracle (E-AURC) **0.026** — E-AURC isolates how well confidence *ranks* its own errors from how many there are.
