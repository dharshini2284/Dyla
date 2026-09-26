"""A local tool for the part of this project that is actually slow: labelling.

The matcher was the interesting engineering. The bottleneck is getting 130
photographs labelled with a catalogue id and a set of failure conditions, by
hand, without introducing the errors that would quietly corrupt every downstream
number.

So this is not a demo. It is built around the two mistakes that are expensive
here and easy to make in a spreadsheet:

  * **assigning the wrong catalogue id.** The tool runs the matcher on each photo
    and offers its top candidates with thumbnails, so the common case is
    confirming a suggestion rather than typing an id. A wrong id is not a
    labelling typo -- it silently converts an in-catalogue photo into an
    out-of-catalogue one, which is exactly the bug that cost this project two
    evaluation runs.

  * **inconsistent condition vocabulary.** Conditions are checkboxes drawn from
    the single source of truth in `eval/manifest.py`, so a free-typed
    "lowlight" can never reach the CSV.

Deliberately stdlib-only: no FastAPI, no npm. A reviewer running this from a
clean checkout should not have to install a web stack to label photographs.
"""

from __future__ import annotations

import csv
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..eval.manifest import CONDITIONS, HEADER, KINDS, SKU_CONFIDENCE, SPLITS

_STATE: dict = {}
_LOCK = threading.Lock()


def validate_row(body: dict) -> str | None:
    """Return an error message, or None if the row is safe to write.

    Extracted from the request handler so it is testable on its own -- these are
    the checks that stop a labelling session from silently corrupting every
    downstream number, so they deserve tests that exercise the real code rather
    than a re-implementation.
    """
    required = {"photo_id", "filename", "kind", "split"}
    missing = sorted(required - set(body))
    if missing:
        return f"missing {missing}"
    if body["kind"] not in KINDS:
        return f"kind must be one of {sorted(KINDS)}"
    if body["split"] not in SPLITS:
        return f"split must be one of {sorted(SPLITS)}"
    if body.get("sku_confidence", "exact") not in SKU_CONFIDENCE:
        return f"sku_confidence must be one of {sorted(SKU_CONFIDENCE)}"
    bad = [c for c in (body.get("conditions") or "").split("|") if c and c not in CONDITIONS]
    if bad:
        return f"unknown conditions {bad}"
    if body["kind"] == "clean" and body.get("conditions"):
        return "clean controls must have no conditions"
    if body["kind"] == "out_of_catalogue" and body.get("item_id"):
        return "out_of_catalogue rows must have an empty item_id"
    if body["kind"] != "out_of_catalogue" and not str(body.get("item_id", "")).isdigit():
        return "item_id must be an integer"
    return None


def is_servable(path: Path, roots: list[Path]) -> bool:
    """True only for real files under one of `roots`.

    The tool serves local files over HTTP, so this is the traversal guard.
    """
    try:
        rp = Path(path).resolve()
    except OSError:
        return False
    if not rp.is_file():
        return False
    return any(str(rp) == str(r) or str(rp).startswith(str(r) + "/")
               for r in (Path(x).resolve() for x in roots))


def _catalogue_thumb(product_id: int) -> str | None:
    """First downloaded view of a catalogue item, as a servable path."""
    root: Path = _STATE["images"]
    p = root / f"{product_id % 100:02d}" / str(product_id) / "0.jpg"
    return str(p) if p.exists() else None


def _match(photo: Path, k: int = 6) -> list[dict]:
    matcher = _STATE.get("matcher")
    if matcher is None:
        return []
    from PIL import Image
    res = matcher.lookup(Image.open(photo), k=k)
    meta = _STATE["meta"]
    out = []
    for c in res.candidates:
        m = meta.get(c.item_id, {})
        out.append({
            "item_id": c.item_id,
            "score": round(float(c.score), 4),
            "brand": m.get("brand", ""),
            "name": m.get("name", ""),
            "colour": m.get("base_colour", ""),
            "thumb": _catalogue_thumb(c.item_id),
        })
    return out


def _load_rows() -> dict[str, dict]:
    path: Path = _STATE["manifest"]
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["filename"]: r for r in csv.DictReader(fh)}


def _save_row(row: dict) -> None:
    path: Path = _STATE["manifest"]
    with _LOCK:
        rows = _load_rows()
        rows[row["filename"]] = row
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=HEADER)
            w.writeheader()
            for r in rows.values():
                w.writerow({k: r.get(k, "") for k in HEADER})


def _photos() -> list[dict]:
    root: Path = _STATE["photos"]
    labelled = _load_rows()
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".heic"}:
            continue
        rel = str(p.relative_to(root))
        r = labelled.get(rel)
        out.append({
            "filename": rel,
            "path": str(p),
            "labelled": r is not None,
            "row": r,
        })
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the console readable
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/":
            return self._send(200, (Path(__file__).parent / "app.html").read_bytes(),
                              "text/html; charset=utf-8")

        if u.path == "/api/config":
            return self._json({
                "conditions": CONDITIONS,
                "kinds": sorted(KINDS),
                "splits": sorted(SPLITS),
                "sku_confidence": sorted(SKU_CONFIDENCE),
                "manifest": str(_STATE["manifest"]),
                "photo_root": str(_STATE["photos"]),
                "has_matcher": _STATE.get("matcher") is not None,
            })

        if u.path == "/api/photos":
            return self._json(_photos())

        if u.path == "/api/match":
            fn = q.get("filename", [""])[0]
            p = _STATE["photos"] / fn
            if not p.exists():
                return self._json({"error": "not found"}, 404)
            try:
                return self._json({"candidates": _match(p)})
            except Exception as exc:                    # a bad image must not kill the tool
                return self._json({"error": str(exc), "candidates": []})

        if u.path == "/img":
            raw = unquote(q.get("p", [""])[0])
            p = Path(raw)
            # Only ever serve from the two roots this tool was pointed at.
            if not is_servable(p, [_STATE["photos"], _STATE["images"]]):
                return self._json({"error": "forbidden"}, 403)
            rp = p.resolve()
            ctype = mimetypes.guess_type(str(rp))[0] or "application/octet-stream"
            return self._send(200, rp.read_bytes(), ctype)

        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)

        if u.path == "/api/save":
            err = validate_row(body)
            if err:
                return self._json({"error": err}, 400)
            _save_row(body)
            return self._json({"ok": True, "labelled": sum(1 for p in _photos() if p["labelled"])})

        return self._json({"error": "not found"}, 404)


def serve(
    photos: Path,
    manifest: Path,
    images: Path,
    items: Path | None = None,
    index: Path | None = None,
    port: int = 8765,
    matcher=None,
) -> None:
    _STATE.update({"photos": Path(photos), "manifest": Path(manifest),
                   "images": Path(images), "matcher": matcher, "meta": {}})
    if items and Path(items).exists():
        meta = {}
        with Path(items).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    meta[r["product_id"]] = r
        _STATE["meta"] = meta

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    n = len(_photos())
    done = sum(1 for p in _photos() if p["labelled"])
    print(f"\n  labelling {n} photos in {photos}  ({done} already labelled)")
    print(f"  writing   {manifest}")
    print(f"  matcher   {'on -- candidates suggested per photo' if matcher else 'off'}")
    print(f"\n  open http://127.0.0.1:{port}    (ctrl-c to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
