# Deploying the matcher UI

The working tree carries ~30 GB of full-resolution multi-view catalogue images.
None of that is needed to serve. A deployment needs one 200px thumbnail per
indexed item plus the index artefacts.

```bash
python scripts/package_deploy.py          # -> deploy/bundle, 27.5 MB for 3,000 items
docker build -t vpm-ui .
docker run --rm -p 8765:8765 vpm-ui       # http://127.0.0.1:8765/ui
```

## What it costs to run

Measured, not estimated:

| | |
|---|---|
| Index artefacts (index + whitener + distractors + calibrator) | 14 MB |
| 200px thumbnails | 3.9 KB each — 12 MB / 3k items, 146 MB / 36.5k |
| Bundle total (3,000 items) | **27.5 MB** |
| Backbone weights | ~400 MB, baked into the image at build time |
| **Peak RSS** | **1.08 GB** |
| **Inference, 2 vCPU, 2 crops** | **115 ms** |

Two deliberate choices in the `Dockerfile`. Torch is installed from PyTorch's
**CPU index** — the default wheel drags in ~2.5 GB of CUDA that is dead weight on
every free tier, all of which are CPU. And the backbone is downloaded at **build**
time, not first request, because a scale-to-zero platform would otherwise pay
~400 MB on every cold start.

## Where it fits on a free tier

The landscape moved in 2026 and several of the obvious answers are gone:

- **HuggingFace Spaces** — free CPU Basic for *compute* Spaces was withdrawn around
  July 2026; only Static Spaces are free. No longer an option for this.
- **Fly.io**, **Koyeb** — free compute tiers withdrawn.

What still works for a 1.08 GB process:

| target | free allowance | verdict |
|---|---|---|
| **Oracle Cloud Always Free** | 4 ARM Ampere cores, 24 GB RAM, always-on | **Best.** Huge headroom, no cold start. Build with `--platform linux/arm64`. Signup needs a card for verification. |
| **Google Cloud Run** | 180k vCPU-s + 2M req/month, scales to zero | Easiest deploy. Set `--memory 2Gi`. Cold start is the cost: ~15 s to load the model even with weights baked in. |

Oracle's *AMD micro* shape (1 GB RAM) is **too small** — we need 1.08 GB.

### Google Cloud Run

```bash
gcloud run deploy vpm-ui --source . \
  --memory 2Gi --cpu 2 --timeout 120 \
  --allow-unauthenticated --region us-central1
```

Cloud Run injects `$PORT`; the image already honours it.

### Oracle Always Free (ARM)

```bash
docker build --platform linux/arm64 -t vpm-ui .
# push to a registry, then on the instance:
docker run -d -p 80:8765 --restart unless-stopped vpm-ui
```

## Getting under 512 MB

If a smaller tier matters, the 1.08 GB is mostly PyTorch itself, not the model.
Exporting the backbone to **ONNX Runtime** would drop the runtime to roughly
150 MB plus weights and remove the torch dependency entirely. That is real work
and is not done here — `REPORT.md` §7 covers the quantisation tradeoffs, and the
honest note is that int8 ViT quantisation damages the *score margin* more than
top-1 accuracy, which matters because the refusal decision lives on that margin.

## Before putting it on a public URL

The catalogue is scraped product data belonging to someone else. Serving it from
a private demo URL you hand a reviewer is one thing; publishing it with their
product imagery on your own host is another. Hotlinking their CDN instead would
cut the bundle to ~14 MB, but leans on their bandwidth and breaks under referrer
checks.

There is also **no authentication** on this server. `--host` defaults to
loopback for that reason; anything public should sit behind the platform's own
auth (Cloud Run's `--no-allow-unauthenticated`, or a reverse proxy).

## Status

The `Dockerfile` is written but **has not been built or run here** — the Docker
daemon was not running on the build machine. The bundle step, the `--host` flag
and the server itself are all verified; the image is not.
