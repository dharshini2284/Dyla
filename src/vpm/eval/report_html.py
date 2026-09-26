"""A self-contained HTML error report: the failure grid the tables cannot show.

`report.md` already carries every number. This exists for the one thing a table
cannot do -- put the query, the answer the system gave, and the answer it should
have given side by side, so a reader can see *why* a case was hard instead of
reading that it was.

Thumbnails are base64-embedded so the file is a single artefact that survives
being emailed or opened from anywhere, with no asset directory to lose.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

THUMB = 190


def _thumb(path: Path, size: int = THUMB) -> str | None:
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _cat_thumb(images_root: Path, pid: int) -> str | None:
    p = images_root / f"{pid % 100:02d}" / str(pid) / "0.jpg"
    return _thumb(p) if p.exists() else None


def _bar(frac: float, label: str) -> str:
    pct = max(0.0, min(1.0, frac)) * 100
    return (f'<div class="meter"><i style="width:{pct:.1f}%"></i></div>'
            f'<span class="mv">{label}</span>')


def build(
    report: dict,
    per_photo: list[dict],
    photo_root: Path,
    images_root: Path,
    items_meta: dict,
    out: Path,
    max_cases: int = 60,
    backbone=None,
) -> Path:
    hard = [r for r in per_photo if r.get("kind") == "hard" and r.get("split") == report.get("split")]
    fails = [r for r in hard if not r.get("correct_sku")]
    wins = [r for r in hard if r.get("correct_sku")]

    def meta(pid):
        m = items_meta.get(pid, {})
        return f"{m.get('brand','')} — {m.get('name','')}".strip(" —") or str(pid)

    cards = []
    def _saliency(path: Path) -> str | None:
        """Overlay the patch-norm map the localiser reads.

        Shown for failures because it answers a question the ranked list cannot:
        was the photo hard, or was the model looking at the wrong part of it?
        """
        if backbone is None:
            return None
        try:
            from ..embed.saliency import heat_overlay, norm_map, salient_box
            im = Image.open(path).convert("RGB")
            heat = norm_map(backbone, im)
            ov = heat_overlay(im, heat, salient_box(heat), max_side=THUMB)
            buf = io.BytesIO()
            ov.save(buf, "JPEG", quality=78)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    for r in (fails[:max_cases]):
        q = _thumb(photo_root / r["filename"])
        sal = _saliency(photo_root / r["filename"])
        got = _cat_thumb(images_root, r["top1"]) if r.get("top1") else None
        want = _cat_thumb(images_root, r["item_id"]) if r.get("item_id") else None
        conds = "".join('<span class="tag">{}</span>'.format(c) for c in r.get("conditions", []))
        rank = r.get("rank_of_truth")
        score_s = "{:.3f}".format(r.get("score", 0) or 0)
        rank_s = str(rank) if rank else "—"
        pm_s = " · p={:.3f}".format(r["p_match"]) if r.get("p_match") is not None else ""
        ref_s = " · REFUSED" if r.get("refused") else ""
        cards.append(f"""
        <article class="case">
          <div class="cls {r.get('error_class','')}">{r.get('error_class','')}</div>
          <div class="trio{' quad' if sal else ''}">
            <figure><img src="{q or ''}" alt=""><figcaption>query</figcaption></figure>
            {f'<figure><img src="{sal}" alt=""><figcaption>saliency</figcaption></figure>' if sal else ''}
            <figure><img src="{got or ''}" alt=""><figcaption>returned</figcaption></figure>
            <figure><img src="{want or ''}" alt=""><figcaption>correct</figcaption></figure>
          </div>
          <div class="meta">
            <div><b>returned</b> {meta(r.get('top1'))}</div>
            <div><b>correct</b> {meta(r.get('item_id'))}</div>
            <div class="sub">score {score_s} · truth at rank {rank_s}{pm_s}{ref_s}</div>
            <div class="tags">{conds}</div>
          </div>
        </article>""")

    per_cond = report.get("per_condition", {})
    cond_parts = []
    for cname, d in sorted(per_cond.items(), key=lambda kv: kv[1]["acc_sku"]):
        ci = "[{:.2f}, {:.2f}]".format(d["wilson_lo"], d["wilson_hi"])
        cond_parts.append(
            "<tr><td>{}</td><td class='n'>{}</td><td class='n'>{:.3f}</td><td>{}</td></tr>".format(
                cname, d["n"], d["acc_sku"], _bar(d["acc_sku"], ci)))
    cond_rows = "".join(cond_parts)

    tax = report.get("taxonomy_hard", {})
    tax_parts = []
    for cname, n in tax.get("counts", {}).items():
        if not n:
            continue
        share = tax["share_of_all"][cname]
        tax_parts.append(
            "<tr><td>{}</td><td class='n'>{}</td><td>{}</td></tr>".format(
                cname, n, _bar(share, "{:.1%}".format(share))))
    tax_rows = "".join(tax_parts)

    me = report.get("marginal_effects", [])
    me_parts = []
    for m in sorted(me, key=lambda d: d["marginal_pp"]):
        cls = "n neg" if m["marginal_pp"] < 0 else "n"
        me_parts.append(
            "<tr><td>{}</td><td class='n'>{}</td><td class='{}'>{:+.1f} pp</td>"
            "<td class='n sub'>[{:+.2f}, {:+.2f}]</td></tr>".format(
                m["condition"], m["n"], cls, m["marginal_pp"], m["lo"], m["hi"]))
    me_rows = "".join(me_parts)

    h = report.get("hard", {})
    c = report.get("clean", {})
    gap = report.get("gap_paired", {})
    ref = report.get("refusal", {})
    tw = ref.get("three_way", {})

    def stat(v, lo=None, hi=None):
        s = f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        return s + (f'<span class="ci">[{lo:.3f}, {hi:.3f}]</span>' if lo is not None else "")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Error Analysis</title>
<style>
:root{{--bg:#fbfbfa;--panel:#fff;--line:#e4e2dd;--ink:#1a1917;--muted:#6b6862;
--accent:#3b5bdb;--ok:#2f7a4d;--bad:#b3261e;--warn:#b4531a;--r:10px}}
:root:not([data-theme="light"]){{@media(prefers-color-scheme:dark){{
--bg:#17171a;--panel:#1f1f23;--line:#33333a;--ink:#ececef;--muted:#9a9aa3;
--accent:#7d95f5;--ok:#6cc48d;--bad:#e5766c;--warn:#e0925a}}}}
:root[data-theme="dark"]{{--bg:#17171a;--panel:#1f1f23;--line:#33333a;--ink:#ececef;
--muted:#9a9aa3;--accent:#7d95f5;--ok:#6cc48d;--bad:#e5766c;--warn:#e0925a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
.page{{max-width:1080px;margin:0 auto;padding:36px 20px 80px}}
h1{{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}}
.lede{{color:var(--muted);margin:0 0 28px;max-width:62ch}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
margin:36px 0 12px;font-weight:650}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:14px}}
.kpi .k{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
.kpi .v{{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.02em}}
.ci{{display:block;font-size:11px;color:var(--muted);font-weight:400}}
table{{width:100%;border-collapse:collapse;background:var(--panel);
border:1px solid var(--line);border-radius:var(--r);overflow:hidden}}
th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);font-size:14px}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}}
tr:last-child td{{border-bottom:none}}
td.n{{font-variant-numeric:tabular-nums}} td.neg{{color:var(--bad)}} td.sub{{color:var(--muted);font-size:12px}}
.meter{{display:inline-block;width:120px;height:6px;background:var(--line);
border-radius:99px;overflow:hidden;vertical-align:middle;margin-right:8px}}
.meter i{{display:block;height:100%;background:var(--accent)}}
.mv{{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}}
.note{{background:var(--panel);border-left:3px solid var(--warn);border-radius:0 8px 8px 0;
padding:10px 14px;color:var(--muted);font-size:13px;margin:12px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px}}
.case{{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);
padding:12px;overflow:hidden}}
.cls{{font-size:11px;font-weight:650;text-transform:uppercase;letter-spacing:.05em;
color:var(--warn);margin-bottom:8px}}
.cls.catastrophic{{color:var(--bad)}} .cls.colourway_confusion{{color:var(--accent)}}
.trio{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}}
.trio.quad{{grid-template-columns:repeat(4,1fr)}}
.trio figure{{margin:0;text-align:center}}
.trio img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:var(--line)}}
.trio figcaption{{font-size:10px;color:var(--muted);margin-top:3px;text-transform:uppercase;
letter-spacing:.05em}}
.meta{{margin-top:10px;font-size:12px}} .meta b{{color:var(--muted);font-weight:600}}
.meta .sub{{color:var(--muted);margin-top:4px;font-variant-numeric:tabular-nums}}
.tags{{margin-top:7px;display:flex;flex-wrap:wrap;gap:4px}}
.tag{{font-size:10px;padding:2px 7px;border:1px solid var(--line);border-radius:99px;color:var(--muted)}}
@media(max-width:680px){{.page{{padding:20px 16px 60px}}}}
</style></head><body><div class="page">

<h1>Error analysis</h1>
<p class="lede">Every failure on the hard set, with the query, the answer given and the
answer expected. Generated by <code>vpm eval</code> — the tables here are the same
numbers as <code>report.md</code>, and the grid is the part a table cannot show.
Where a <em>saliency</em> panel is present it is the patch-token norm map the
localiser reads (warm = high norm, box = the crop it would take); it answers
whether a case failed because the photo was hard or because the model was looking
at the wrong part of it.</p>

<div class="kpis">
  <div class="kpi"><div class="k">hard R@1</div><div class="v">{stat(h.get('r1_sku',{}).get('point',0), h.get('r1_sku',{}).get('lo'), h.get('r1_sku',{}).get('hi'))}</div></div>
  <div class="kpi"><div class="k">clean R@1</div><div class="v">{stat(c.get('r1_sku',{}).get('point',0))}</div></div>
  <div class="kpi"><div class="k">paired gap</div><div class="v">{gap.get('delta',0)*100:+.1f} pp<span class="ci">p = {gap.get('p_value',float('nan')):.2g}</span></div></div>
  <div class="kpi"><div class="k">refusal AUROC</div><div class="v">{ref.get('auroc',float('nan')):.3f}</div></div>
  <div class="kpi"><div class="k">false accept</div><div class="v">{tw.get('ooc_accepted_FAR',float('nan')):.3f}</div></div>
  <div class="kpi"><div class="k">wrong accept</div><div class="v">{tw.get('wrong_accept_rate',float('nan')):.3f}</div></div>
</div>
<div class="note">Clean R@1 is an artefact when run on the synthetic stand-in set — the
clean control <em>is</em> the indexed image, so it matches exactly by construction.
It is a wiring check, not a result.</div>

<h2>Error taxonomy</h2>
<table><tr><th>class</th><th>n</th><th>share of all hard photos</th></tr>{tax_rows}</table>

<h2>Accuracy by failure condition</h2>
<table><tr><th>condition</th><th>n</th><th>R@1</th><th>95% Wilson</th></tr>{cond_rows}</table>
<div class="note">Cells are small — at n=10 a 95% interval spans roughly ±25 pp — so the
<strong>ordering</strong> of these conditions is not resolvable from this set. Read the
ranking from the marginal effects below instead.</div>

<h2>Marginal effect of each condition</h2>
<table><tr><th>condition</th><th>n</th><th>effect</th><th>95% CI (log-odds)</th></tr>{me_rows}</table>
<div class="note">Conditions co-occur, so per-condition accuracy above is confounded.
These are penalised logistic coefficients bootstrapped by item — exploratory at this
sample size. Only intervals excluding zero should be read as real.</div>

<h2>Failures — {len(fails)} of {len(hard)} hard photos{' (showing first ' + str(max_cases) + ')' if len(fails) > max_cases else ''}</h2>
<div class="grid">{''.join(cards)}</div>

<p class="lede" style="margin-top:32px">{len(wins)} correct, {len(fails)} wrong.</p>
</div></body></html>""", encoding="utf-8")
    return out
