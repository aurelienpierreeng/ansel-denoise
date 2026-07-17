"""Crawl the PlayRaw category of discuss.pixls.us into training shards.

PlayRaw (~2000 topics) is people sharing their own raw files for others to
edit — real photographs with declared Creative Commons licenses, which makes
it the content-diversity complement to the raw.pixls.us decoder-coverage
archive. This crawler walks the category through Discourse's JSON API (the
sitemap lists every forum topic unscoped, so the category feed is the better
enumerator), reads each topic's first post, verifies the declared license,
downloads the attached raws and funnels them through the exact same gate ->
decode -> tile pipeline as the archive harvester. Ledger, shard format and
resume semantics are identical; provenance (topic URL, author, license) is
recorded in both the ledger and the shard.

Licenses: only topics declaring CC0 or CC-BY are used by default (attribution
data is retained). Share-alike and non-commercial variants are excluded
unless explicitly allowed with --licenses.

Usage:
    python -m ansel_denoise.crawl_playraw --out shards/playraw
    python -m ansel_denoise.crawl_playraw --out shards/playraw --limit 20   # trial
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

from .harvest import EXCLUDE_EXTENSIONS, RAW_EXTENSIONS, _pack_worker, read_exif, run_isolated

USER_AGENT = "ansel-denoise-harvester/0.1 (https://github.com/aurelienpierreeng/ansel-denoise)"
POLITE_DELAY_S = 0.5  # between HTTP requests to the forum


# ---------------------------------------------------------------------------
# pure functions (unit-tested offline)
# ---------------------------------------------------------------------------

def detect_license(cooked: str) -> str | None:
    """Return a normalized license tag from a post's HTML, or None.

    The most reliable signal is a creativecommons.org link; free-text
    declarations ("This file is licensed Creative Commons, By-Attribution,
    Share-Alike") are the fallback.
    """
    m = re.search(r"creativecommons\.org/(?:licenses/([a-z-]+)|publicdomain/(?:zero|mark))",
                  cooked, re.I)
    if m:
        return (m.group(1) or "cc0").lower()
    text = re.sub(r"<[^>]+>", " ", cooked)
    if re.search(r"(?i)\bcc0\b|public domain", text):
        return "cc0"
    if re.search(r"(?i)creative\s*commons|\bcc[- ]by", text):
        nc = bool(re.search(r"(?i)non[- ]?commercial|\bnc\b", text))
        sa = bool(re.search(r"(?i)share[- ]?alike|\bsa\b", text))
        return "by" + ("-nc" if nc else "") + ("-sa" if sa else "")
    return None


def extract_raw_links(cooked: str, base_url: str) -> list[str]:
    """Absolute URLs of raw-file attachments/links in a post's HTML."""
    links = []
    for m in re.finditer(r'href="([^"]+)"', cooked):
        href = m.group(1)
        stem = href.split("?")[0]
        ext = ("." + stem.rsplit(".", 1)[-1]).lower() if "." in stem.rsplit("/", 1)[-1] else ""
        if ext in RAW_EXTENSIONS and ext not in EXCLUDE_EXTENSIONS:
            links.append(urljoin(base_url, href))
    return list(dict.fromkeys(links))  # dedupe, keep order


# ---------------------------------------------------------------------------
# forum API
# ---------------------------------------------------------------------------

def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def get_json(url: str) -> dict:
    time.sleep(POLITE_DELAY_S)
    return json.loads(_get(url))


def resolve_category(forum: str, slug: str) -> int:
    site = get_json(f"{forum}/site.json")
    for c in site["categories"]:
        if c["slug"] == slug:
            return c["id"]
    raise SystemExit(f"category '{slug}' not found on {forum}")


def list_topics(forum: str, slug: str, cid: int, limit: int = 0) -> list[dict]:
    """All topics of the category, newest first: [{id, title}, ...]."""
    topics, page = [], 0
    while True:
        d = get_json(f"{forum}/c/{slug}/{cid}.json?page={page}")
        batch = d["topic_list"]["topics"]
        if not batch:
            break
        topics.extend({"id": t["id"], "title": t["title"]} for t in batch)
        print(f"  listed {len(topics)} topics...", end="\r", flush=True)
        if limit and len(topics) >= limit:
            return topics[:limit]
        if "more_topics_url" not in d["topic_list"]:
            break
        page += 1
    return topics


def first_post(forum: str, tid: int) -> dict:
    d = get_json(f"{forum}/t/{tid}.json")
    p0 = d["post_stream"]["posts"][0]
    return {"cooked": p0["cooked"], "author": p0.get("username", "?")}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def crawl_topic(topic: dict, forum: str, out_dir: Path, args: argparse.Namespace,
                allowed: set[str]) -> list[dict]:
    """Process one topic; returns one ledger record per raw link (possibly [])."""
    tid = topic["id"]
    post = first_post(forum, tid)
    lic = detect_license(post["cooked"])
    base = {"topic": f"{forum}/t/{tid}", "author": post["author"], "license": lic}

    if lic not in allowed:
        return [{**base, "path": f"playraw/{tid}", "status": "rejected",
                 "reason": f"license {lic or 'undeclared'} not in {sorted(allowed)}"}]

    records = []
    for url in extract_raw_links(post["cooked"], forum):
        fname = url.split("?")[0].rsplit("/", 1)[-1]
        rel = f"playraw/{tid}/{fname}"
        record = {**base, "path": rel, "url": url, "status": "rejected"}
        tmp = out_dir / ".download" / fname
        tmp.parent.mkdir(exist_ok=True)
        try:
            time.sleep(POLITE_DELAY_S)
            tmp.write_bytes(_get(url))
        except Exception as e:
            records.append({**record, "status": "error", "reason": f"download failed: {e!r}"[:300]})
            continue
        try:
            exif = read_exif(tmp)
            iso = exif.get("ISO")
            if iso is None or (isinstance(iso, str) and not iso.isdigit()):
                records.append({**record, "reason": "no ISO in EXIF"})
                continue
            iso = int(iso)
            record["iso"] = iso
            record["camera"] = f"{exif.get('Make', '?')} {exif.get('Model', '?')}"
            if iso > args.max_iso:
                records.append({**record, "reason": f"ISO {iso} > {args.max_iso}"})
                continue
            result = run_isolated(
                _pack_worker,
                (str(tmp), rel, str(out_dir), args.tile_size, args.tiles, args.seed,
                 iso, record["camera"], url),
            )
            records.append({**record, **result})
        finally:
            tmp.unlink(missing_ok=True)
    if not records:
        records.append({**base, "path": f"playraw/{tid}", "status": "rejected",
                        "reason": "no raw links in first post"})
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="shard output directory")
    ap.add_argument("--forum", default="https://discuss.pixls.us")
    ap.add_argument("--category", default="playraw", help="Discourse category slug")
    ap.add_argument("--licenses", default="cc0,by",
                    help="comma list of accepted license tags (default cc0,by)")
    ap.add_argument("--max-iso", type=int, default=200, help="reject files above this ISO")
    ap.add_argument("--tiles", type=int, default=16, help="max tiles per source file")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="stop after N topics (0 = all)")
    ap.add_argument("--seed", type=int, default=0xA45E1)
    args = ap.parse_args(argv)

    allowed = {t.strip() for t in args.licenses.split(",") if t.strip()}
    args.out.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out / "ledger.jsonl"
    done = set()
    if ledger_path.exists():
        with open(ledger_path, encoding="utf-8") as f:
            done = {json.loads(line)["path"] for line in f if line.strip()}
    # a whole topic is done when any of its paths are ledgered
    done_topics = {p.split("/")[1] for p in done if p.startswith("playraw/")}

    cid = resolve_category(args.forum, args.category)
    topics = [t for t in list_topics(args.forum, args.category, cid, args.limit)
              if str(t["id"]) not in done_topics]
    print(f"{len(topics)} topics to process ({len(done_topics)} already in ledger)")

    n_ok = 0
    with open(ledger_path, "a", encoding="utf-8") as ledger:
        for i, topic in enumerate(topics):
            try:
                records = crawl_topic(topic, args.forum, args.out, args, allowed)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # one bad topic must not kill the crawl
                records = [{"path": f"playraw/{topic['id']}", "status": "error",
                            "reason": repr(e)[:300]}]
            for record in records:
                ledger.write(json.dumps(record) + "\n")
                n_ok += record["status"] == "harvested"
            ledger.flush()
            summary = "; ".join(f"{r['status']}" + (f" ({r.get('reason', '')})"
                                if r["status"] != "harvested" else f" {r.get('n_tiles')} tiles")
                                for r in records)
            print(f"[{i + 1}/{len(topics)}] t/{topic['id']} {topic['title'][:40]!r}: {summary}")
    print(f"done: {n_ok} raws harvested into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
