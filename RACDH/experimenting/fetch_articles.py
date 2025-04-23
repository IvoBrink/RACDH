#!/usr/bin/env python3
"""
Random Wikipedia Snippet Fetcher ≥ `MIN_CHARS` characters (fully self‑validating)
==========================================================================
Maintains `random_wiki_articles.ndjson` with **exactly `TARGET` snippets**, each at
least `MIN_CHARS` visible characters after whitespace‑normalisation.

Why this version?
-----------------
The previous iteration relied on a simple length check, but users reported
snippets < `MIN_CHARS` still slipping through.  Causes include leading/trailing
newlines, multiple spaces, and HTML entities that collapse when rendered.

**Fixes added**
* Strict length check on a *normalised* string: collapse all runs of
  whitespace → single spaces, then `strip()`.
* Post‑run **validation pass** that aborts (non‑zero exit) if any snippet fails
  the length rule, so you’ll know immediately.
* Optional `--revalidate` CLI flag to *only* clean & validate an existing file—no
  network traffic.

Usage examples
--------------
```bash
# Default behaviour: clean existing, fetch new pages until TARGET reached
python random_wiki_fetcher_clean.py

# Only revalidate & prune shorts; do NOT hit the API\python random_wiki_fetcher_clean.py --revalidate

# Custom size & target
python random_wiki_fetcher_clean.py --min 400 --target 25000
```

Dependencies: `requests`, `tqdm`, `typer` (for neat CLI). Install via:
```bash
pip install requests tqdm typer[all]
```
"""

from __future__ import annotations

import json
import time
import pathlib
import re
import sys
from typing import List, Dict, Tuple, Set

import requests
from tqdm import tqdm
import typer

# ─── CONFIG (default values) ──────────────────────────────────────────────────
DEFAULT_TARGET = 20_000    # how many snippets to end up with
DEFAULT_MIN    = 300       # min chars after whitespace normalisation
BATCH_SIZE     = 500       # 500 for normal, 5000 for bot accounts
SLEEP_SECONDS  = 0.5       # throttle so we stay polite
OUTFILE        = "random_wiki_articles.ndjson"

API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
USER_AGENT   = "RandomWikiFetcher/0.4 (youremail@example.com)"

# ─── UTILS ────────────────────────────────────────────────────────────────────
_ws_re = re.compile(r"\s+")

def normalise(text: str) -> str:
    """Collapse all whitespace to single spaces and strip ends."""
    return _ws_re.sub(" ", text).strip()


def load_existing(path: str, min_chars: int) -> Tuple[Set[int], List[Dict]]:
    """Return (seen_ids, good_records) that satisfy length ≥ *min_chars*."""
    seen: Set[int] = set()
    good: List[Dict] = []
    p = pathlib.Path(path)
    if not p.exists():
        return seen, good

    with p.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("pageid")
            text = normalise(rec.get("text") or "")
            if len(text) >= min_chars and pid not in seen:
                rec["text"] = text  # ensure text is normalised
                good.append(rec)
                seen.add(pid)
    return seen, good


def atomic_write(path: str, records: List[Dict]):
    tmp = pathlib.Path(path).with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def validate(records: List[Dict], min_chars: int) -> bool:
    """Return True if all records meet length threshold, else print issues."""
    short = [r for r in records if len(r["text"]) < min_chars]
    if not short:
        return True
    print("\nValidation failed — short snippets found:\n")
    for r in short[:10]:  # show up to 10 examples
        print(f" • {r['title']!r} ({len(r['text'])} chars)")
    print(f"…and {len(short)-10} more." if len(short) > 10 else "")
    return False

# ─── MAIN FETCH LOGIC ─────────────────────────────────────────────────────────

def fetch_snippets(target: int, min_chars: int, revalidate_only: bool):
    seen_ids, records = load_existing(OUTFILE, min_chars)
    print(f"Loaded {len(records):,} records ≥{min_chars} chars; need {target - len(records):,} more.")

    if revalidate_only:
        ok = validate(records, min_chars)
        if ok:
            print("All records meet the length requirement. ✅")
            sys.exit(0)
        else:
            # Rewrite without the short ones and exit with error
            atomic_write(OUTFILE, records)
            sys.exit(1)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnfilterredir": "nonredirects",
        "grnlimit": BATCH_SIZE,
        "prop": "extracts",
        "explaintext": 1,
        "exintro": 1,
        "exsentences": 10,
        "exlimit": "max",
    }

    pbar = tqdm(total=target, initial=len(records), unit="article")
    try:
        while len(records) < target:
            response = session.get(API_ENDPOINT, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for page in data.get("query", {}).get("pages", {}).values():
                pid = page["pageid"]
                if pid in seen_ids:
                    continue
                text = normalise(page.get("extract") or "")
                if len(text) < min_chars:
                    continue
                records.append({
                    "pageid": pid,
                    "title" : page["title"],
                    "text"  : text,
                })
                seen_ids.add(pid)
                pbar.update(1)
                if len(records) >= target:
                    break

            if "continue" in data:
                params.update(data["continue"])
            else:
                params = {k: v for k, v in params.items() if not k.endswith("continue")}
            time.sleep(SLEEP_SECONDS)
    finally:
        pbar.close()

    print("Validating final file…")
    if not validate(records, min_chars):
        print("Writing cleaned subset (without shorts) and aborting with error…")
        atomic_write(OUTFILE, records)
        sys.exit(1)

    print("Writing… all records valid. ✅")
    atomic_write(OUTFILE, records)
    print(f"Done — {len(records):,} records ≥{min_chars} chars saved to {OUTFILE}.")

# ─── CLI ENTRYPOINT ──────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False, help="Random Wikipedia snippet fetcher with strict length filter.")

@app.command()
def main(
    target: int = typer.Option(DEFAULT_TARGET, "--target", help="Number of snippets to keep."),
    min:    int = typer.Option(DEFAULT_MIN,    "--min",    help="Minimum characters per snippet."),
    revalidate: bool = typer.Option(False, "--revalidate", help="Only clean/validate existing file without fetching."),
):
    """Fetch up to TARGET random Wikipedia intros ≥ MIN characters each."""
    fetch_snippets(target, min, revalidate)


if __name__ == "__main__":
    app()