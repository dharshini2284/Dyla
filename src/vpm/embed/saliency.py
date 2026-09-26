"""Localise the object in a cluttered query photo using patch-token norms.

The single biggest accuracy lever against the domain gap, and it is nearly free:
we are already paying one forward pass, and the patch tokens it produces carry a
usable foreground signal. Background patches have systematically lower feature
norm than object patches, so thresholding the norm map yields a rough object
mask with no extra model, no extra download, and no segmentation dependency.

Compared to the alternatives: `rembg`/ISNet gives a cleaner mask but costs
hundreds of milliseconds, which breaks any interactive budget, and it can cut the
shoe when a hand is in frame -- one of the exact failure conditions under test.
This costs one extra forward pass (11-36 ms) and degrades gracefully.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from .backbone import Backbone


def norm_map(backbone: Backbone, img: Image.Image) -> np.ndarray:
    """Per-patch L2 norm, reshaped to the patch grid and scaled to [0, 1]."""
    batch = backbone.transform(img.convert("RGB")).unsqueeze(0)
    tok = backbone.tokens(batch)[0]                 # (n_patches, D)
    g = backbone.grid()
    n = g * g
    if tok.shape[0] < n:
        return np.ones((g, g), dtype=np.float32)
    mags = torch.linalg.vector_norm(tok[:n], dim=-1).reshape(g, g).numpy()
    lo, hi = float(mags.min()), float(mags.max())
    if hi - lo < 1e-6:
        return np.ones((g, g), dtype=np.float32)
    return ((mags - lo) / (hi - lo)).astype(np.float32)


def salient_box(
    heat: np.ndarray,
    quantile: float = 0.6,
    margin: float = 0.12,
    largest_component: bool = True,
) -> tuple[float, float, float, float]:
    """Relative (l, t, r, b) box around the salient region, with a margin.

    Takes the bounding box of the LARGEST CONNECTED COMPONENT rather than of all
    above-threshold patches. Without registers, DINOv2 emits a handful of
    high-norm artefact tokens scattered across background regions; measured on
    our own catalogue images, plain ViT-S/14 produced 4 components where 9 stray
    patches stretched the box from rows 4-12 to rows 1-14, while the `reg4`
    variant produced a single clean component. Registers are the real fix and are
    the default backbone for localisation; this is the belt-and-braces guard for
    when they are unavailable.

    Falls back to the full frame when the heat map has no structure, so a failed
    localisation degrades to the no-crop baseline rather than to a wrong crop.
    """
    thr = float(np.quantile(heat, quantile))
    mask = heat >= thr
    if not mask.any():
        return 0.0, 0.0, 1.0, 1.0
    if largest_component:
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        if n > 2:  # more than one blob (label 0 is background)
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            mask = labels == int(np.argmax(counts))
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    g = heat.shape[0]
    t, b = rows[0] / g, (rows[-1] + 1) / g
    l, r = cols[0] / g, (cols[-1] + 1) / g
    mw, mh = (r - l) * margin, (b - t) * margin
    return (
        float(max(0.0, l - mw)),
        float(max(0.0, t - mh)),
        float(min(1.0, r + mw)),
        float(min(1.0, b + mh)),
    )


def crop_to_box(img: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    l, t, r, b = box
    W, H = img.size
    px = (int(l * W), int(t * H), max(int(r * W), int(l * W) + 1), max(int(b * H), int(t * H) + 1))
    return img.crop(px)


def heat_overlay(
    img: Image.Image,
    heat: np.ndarray,
    box: tuple[float, float, float, float] | None = None,
    alpha: float = 0.55,
    max_side: int = 420,
) -> Image.Image:
    """Render the patch-norm map over the photo, with the chosen crop box.

    This visualises the signal the localiser ACTUALLY uses -- per-patch feature
    norm -- rather than a Grad-CAM or attention map of something the system does
    not consult. Showing a different saliency than the one driving the crop would
    be a prettier picture and a false explanation.

    Warm = high norm = foreground. The drawn rectangle is the box that was fed to
    the backbone, so a wrong crop is visible rather than inferred.
    """
    import cv2

    rgb = img.convert("RGB")
    rgb.thumbnail((max_side, max_side), Image.LANCZOS)
    W, H = rgb.size
    base = np.asarray(rgb).astype(np.float32)

    h = cv2.resize(heat.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    h = np.clip((h - h.min()) / (float(h.max() - h.min()) + 1e-9), 0, 1)
    # Blue (cold, background) -> red (warm, object). Built by hand so the module
    # keeps no matplotlib dependency.
    cmap = np.stack([h, np.clip(1.2 * (1 - np.abs(h - 0.5) * 2), 0, 1), 1 - h], axis=-1) * 255.0

    out = (1 - alpha) * base + alpha * cmap
    out = np.clip(out, 0, 255).astype(np.uint8)

    if box is not None:
        l, t, r, b = box
        cv2.rectangle(out, (int(l * W), int(t * H)), (int(r * W) - 1, int(b * H) - 1),
                      (255, 255, 255), 2)
    return Image.fromarray(out)


def localise(
    backbone: Backbone, img: Image.Image, quantile: float = 0.6, margin: float = 0.12
) -> tuple[Image.Image, tuple[float, float, float, float], float]:
    """Return (cropped image, box, salient area fraction).

    The area fraction is kept because it doubles as a confidence feature: a
    localisation covering 99% or 2% of the frame usually means it failed.
    """
    heat = norm_map(backbone, img)
    box = salient_box(heat, quantile, margin)
    area = (box[2] - box[0]) * (box[3] - box[1])
    return crop_to_box(img, box), box, float(area)
