"""Re-ranking and query expansion. Both expansions default to OFF, on evidence.

Two techniques that the closed-set retrieval literature reports as near-universal
wins are actively harmful here, and the reason is specific to open-set matching
over a catalogue full of colourway variants:

**alpha query expansion** blends the query with its top neighbours. For an
out-of-catalogue photo that drags the query TOWARD the catalogue manifold,
mechanically inflating its top-1 similarity. It is a technique for making
everything look more like a match, so it raises exactly the false-accept rate the
refusal work is graded on. If used at all, the confidence features must be
computed on the PRE-expansion query.

**database-side augmentation** smooths each catalogue vector toward its
neighbours. Our hardest negatives are near-identical SKUs differing only in
colourway -- which are precisely each other's nearest neighbours -- so DBA
averages a colourway cluster toward its centroid and destroys the distinction
that matters most. Expect it to help coarse style accuracy and hurt exact-SKU
top-1.

**Geometric verification** is here as a CONFIDENCE FEATURE, not a re-ranker. The
evidence it provides is one-sided: 40 RANSAC inliers is near-proof of a match,
while 2 inliers means "low texture", not "not a match". That asymmetry makes it
useless for ordering the low-texture majority and useful as one input among many
to a calibrated model that has other features to fall back on.
"""

from __future__ import annotations

import numpy as np


def alpha_qe(
    q: np.ndarray,
    neighbour_emb: np.ndarray,
    neighbour_scores: np.ndarray,
    alpha: float = 3.0,
    k: int = 5,
) -> np.ndarray:
    """Weighted blend of the query with its top-k neighbours.

    Off by default -- see the module docstring for why this raises FAR.
    """
    if len(neighbour_emb) == 0:
        return q
    k = min(k, len(neighbour_emb))
    w = np.maximum(neighbour_scores[:k], 0.0) ** alpha
    if w.sum() <= 0:
        return q
    expanded = q + (neighbour_emb[:k] * w[:, None]).sum(axis=0) / w.sum()
    return expanded / (np.linalg.norm(expanded) + 1e-12)


def database_augmentation(emb: np.ndarray, k: int = 3, weight: float = 0.5) -> np.ndarray:
    """Smooth each catalogue vector toward its k nearest neighbours.

    Off by default -- it averages colourway clusters toward their centroid.
    O(N^2); intended for ablation on a subsample, not production indexing.
    """
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    neigh = emb[idx].mean(axis=1)
    out = emb + weight * neigh
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def sift_inliers(
    query_img,
    candidate_img,
    max_side: int = 320,
    ratio: float = 0.75,
    ransac_thresh: float = 6.0,
) -> tuple[int, float]:
    """RANSAC inlier count between two images, as a one-sided confidence feature.

    Returns (inliers, inlier_ratio). Downscales first -- SIFT cost is superlinear
    in pixels and the discriminative marks on a shoe survive 320px.

    An affine model is used rather than a homography: a shoe upper deforms, so
    the planar assumption a homography makes is wrong and costs inliers on
    genuine matches.
    """
    import cv2

    def prep(im):
        a = np.asarray(im.convert("L"))
        h, w = a.shape
        scale = max_side / max(h, w)
        if scale < 1.0:
            a = cv2.resize(a, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return a

    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(prep(query_img), None)
    k2, d2 = sift.detectAndCompute(prep(candidate_img), None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return 0, 0.0

    matcher = cv2.BFMatcher()
    raw = matcher.knnMatch(d1, d2, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 4:
        return len(good), 0.0

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                           ransacReprojThreshold=ransac_thresh)
    if mask is None:
        return 0, 0.0
    inliers = int(mask.sum())
    return inliers, inliers / max(len(good), 1)
