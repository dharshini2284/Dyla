"""A held-out distractor database, and the score normalisation it enables.

This is the block that breaks the tie raw cosine cannot. Restating the problem:
a dark blurry photo of an in-catalogue shoe scores LOW, and a crisp photo of an
absent shoe that resembles a catalogue one scores HIGH. Both move the same
direction as a threshold slides, so no threshold on top-1 similarity separates
them.

The distractor database supplies the missing axis. It holds footwear that is
deliberately NOT in the catalogue, so for any query we can ask a likelihood-ratio
question rather than an absolute one:

    how much better does the CATALOGUE explain this photo
    than a generic pile of shoes does?

  * dark in-catalogue photo -> low catalogue score, low distractor score  -> ACCEPT
  * crisp absent shoe       -> high catalogue score, high distractor score -> REFUSE

Style clusters, not items, are held out: 36% of individually held-out items keep
a colourway twin in the catalogue, which would put a near-duplicate of a
"distractor" back in the index it is supposed to be independent of.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DistractorDB:
    """Flat bank of embeddings from items guaranteed absent from the catalogue."""

    emb: np.ndarray                 # (N, D) L2-normalised, same space as the index
    dim: int

    def __len__(self) -> int:
        return len(self.emb)

    def scores(self, q: np.ndarray) -> np.ndarray:
        """Similarity of one query against every distractor."""
        if len(self.emb) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.emb @ np.asarray(q, dtype=np.float32).reshape(-1)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, emb=self.emb)

    @staticmethod
    def load(path: Path) -> "DistractorDB":
        z = np.load(path)
        emb = z["emb"].astype(np.float32)
        return DistractorDB(emb=emb, dim=emb.shape[1] if len(emb) else 0)


def build_from_embeddings(emb: np.ndarray) -> DistractorDB:
    e = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
    n = np.linalg.norm(e, axis=1, keepdims=True)
    e = e / np.maximum(n, 1e-12)
    return DistractorDB(emb=e, dim=e.shape[1] if len(e) else 0)
