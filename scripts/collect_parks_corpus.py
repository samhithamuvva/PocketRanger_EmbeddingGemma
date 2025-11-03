#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_parks_corpus.py (policy-aware)
- Fetches NPS API data for given park codes.
- Expands beyond `places` to include policy/FAQ style endpoints (alerts, campgrounds, feespasses, hours, articles, events).
- Normalizes records into long-form text and chunks them (~500 tokens, 100 overlap).
- Writes a clean JSONL corpus for downstream embedding/RAG.

Usage:
  $env:NPS_API_KEY="YOUR_KEY"
  python scripts/collect_parks_corpus.py --parks yose seki grca ^
    --api-endpoints parks alerts places thingstodo campgrounds feespasses hours articles events ^
    --out-root data/raw --clean-out data/clean/parks_corpus.jsonl --page-limit 100
"""
import argparse, os, time, json, pathlib, re
from typing import Dict, Any, Iterable, List, Tuple

import requests

API_BASE = "https://developer.nps.gov/api/v1"

DEFAULT_ENDPOINTS = [
    "parks", "alerts", "places", "thingstodo",
    "campgrounds", "feespasses", "hours", "articles", "events"
]

# ------------------------- utils -------------------------
def ensure_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def chunk_text(tokens: List[str], size: int, overlap: int) -> Iterable[List[str]]:
    step = max(1, size - overlap)
    for i in range(0, len(tokens), step):
        yield tokens[i:i+size]

def basic_clean(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s

def now_iso() -> str:
    import datetime as dt
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ------------------------- API fetch -------------------------
def fetch_nps_api(park_code: str, endpoint: str, api_key: str, out_file: pathlib.Path,
                  page_limit: int = 100, delay_s: float = 0.25) -> int:
    """
    Fetch endpoint pages and append JSON lines `{endpoint, park, data}` to out_file.
    Returns number of items (not pages).
    """
    ensure_dir(out_file.parent)
    headers = {"X-Api-Key": api_key}
    total = 0
    start = 0
    limit = 50  # NPS API typical page size
    pages = 0

    with out_file.open("w", encoding="utf-8") as f:
        while pages < page_limit:
            params = {"parkCode": park_code, "start": start, "limit": limit}
            url = f"{API_BASE}/{endpoint}"
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code != 200:
                print(f"[WARN] {endpoint}/{park_code} status={r.status_code}: {r.text[:200]}")
                break
            payload = r.json()
            data = payload.get("data", [])
            if not data:
                break
            for item in data:
                f.write(json.dumps({"endpoint": endpoint, "park": park_code, "data": item}, ensure_ascii=False) + "\n")
            total += len(data)
            pages += 1
            start += limit
            time.sleep(delay_s)
    return total

# ------------------------- normalize -------------------------
def extract_text_from_item(endpoint: str, d: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """
    Returns (title, section_heading, body_text, url, last_updated)
    Tries to assemble policy-relevant text fields per endpoint.
    """
    data = d.get("data", {})
    # Common fields
    title = data.get("title") or data.get("name") or data.get("fullName") or ""
    url = data.get("url") or data.get("parkUrl") or data.get("permalink") or data.get("listingUrl") or ""
    last_updated = data.get("lastIndexedDate") or data.get("lastIndexedDateTime") or data.get("lastUpdated") or ""

    # Build a readable body by endpoint
    pieces: List[str] = []
    if endpoint in ("parks",):
        pieces += [data.get("description", ""), data.get("directionsInfo", ""), data.get("operatingHours", "")]
    if endpoint in ("alerts",):
        pieces += [data.get("category", ""), data.get("description", ""), data.get("lastIndexedDate", "")]
    if endpoint in ("places",):
        pieces += [data.get("listingDescription", ""), data.get("bodyText", ""), data.get("geojson", "")]
    if endpoint in ("thingstodo",):
        pieces += [data.get("shortDescription", ""), data.get("activityDescription", "")]
    if endpoint in ("articles", "events"):
        pieces += [data.get("listingDescription", ""), data.get("bodyText", "")]
    if endpoint in ("campgrounds",):
        pieces += [
            data.get("description", ""),
            data.get("regulationsoverview", ""),
            data.get("reservationsDescription", ""),
            data.get("reservationsUrl", ""),
            data.get("accessibility", ""),
            data.get("amenities", "")
        ]
    if endpoint in ("feespasses",):
        pieces += [
            data.get("entranceFees", ""), data.get("entrancePasses", ""),
            data.get("fees", ""), data.get("description", "")
        ]
    if endpoint in ("hours",):
        pieces += [
            data.get("standardHours", ""), data.get("description", ""),
            data.get("exceptions", "")
        ]

    body = basic_clean(" ".join([str(x) for x in pieces if x]))
    section_heading = endpoint  # simple label for now
    title = basic_clean(title)
    url = basic_clean(url)
    last_updated = basic_clean(last_updated)
    return title, section_heading, body, url, last_updated

def build_clean_corpus(parks: List[str], raw_root: pathlib.Path, out_clean: pathlib.Path,
                       chunk_target: int = 500, chunk_overlap: int = 100) -> int:
    """
    Reads raw JSONL pages from data/raw/{park}/api/*.jsonl,
    normalizes objects to text, chunks, and writes JSONL lines to out_clean.
    """
    ensure_dir(out_clean.parent)
    n_chunks = 0
    with out_clean.open("w", encoding="utf-8") as out:
        for park in parks:
            api_dir = raw_root / park / "api"
            if not api_dir.exists():
                continue
            for jf in sorted(api_dir.glob("*.jsonl")):
                endpoint = jf.stem
                with jf.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        title, section_heading, body, url, last_updated = extract_text_from_item(endpoint, rec)
                        if not body or len(body) < 120:
                            # tiny descriptions are rarely useful for policy Qs
                            continue
                        # token-ish split
                        toks = body.split()
                        for chunk in chunk_text(toks, size=chunk_target, overlap=chunk_overlap):
                            out.write(json.dumps({
                                "park_code": park,
                                "content_type": endpoint,
                                "title": title,
                                "section_heading": section_heading,
                                "body_text": " ".join(chunk),
                                "url": url,
                                "source": "nps_api",
                                "last_updated": last_updated
                            }, ensure_ascii=False) + "\n")
                            n_chunks += 1
    return n_chunks

# ------------------------- CLI -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parks", nargs="+", required=True, help="e.g. yose seki grca")
    ap.add_argument("--api-endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--out-root", default="data/raw")
    ap.add_argument("--clean-out", default="data/clean/parks_corpus.jsonl")
    ap.add_argument("--page-limit", type=int, default=100)
    ap.add_argument("--api-delay", type=float, default=0.25)
    args = ap.parse_args()

    api_key = os.environ.get("NPS_API_KEY", "")
    if not api_key:
        print("[WARN] NPS_API_KEY not set. Some endpoints may throttle or fail.")

    parks = args.parks
    endpoints = args.api_endpoints
    raw_root = pathlib.Path(args.out_root)
    out_clean = pathlib.Path(args.clean_out)

    # 1) Fetch raw
    total_items = 0
    for park in parks:
        api_dir = ensure_dir(raw_root / park / "api")
        for ep in endpoints:
            out_file = api_dir / f"{ep}.jsonl"
            print(f"[API] Fetching {ep} for {park} → {out_file}")
            count = fetch_nps_api(
                park_code=park, endpoint=ep, api_key=api_key, out_file=out_file,
                page_limit=args.page_limit, delay_s=args.api_delay
            )
            print(f"     {ep}/{park}: {count} items")
            total_items += count

    # 2) Normalize + chunk
    n_chunks = build_clean_corpus(
        parks=parks, raw_root=raw_root, out_clean=out_clean,
        chunk_target=500, chunk_overlap=100
    )

    print(f"[DONE] Raw items: ~{total_items} | Clean chunks: {n_chunks}")
    print(f"[NEXT] Inspect {out_clean} — fields: park_code, content_type, title, section_heading, body_text, url, source, last_updated")

if __name__ == "__main__":
    main()
