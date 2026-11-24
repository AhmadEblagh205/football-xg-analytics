"""Download the StatsBomb open Arsenal 2003/04 league-match sample.

Competition 2 / season 44 contains Arsenal's 38 Premier League matches, not
all 380 league fixtures.  Events and lineups are both downloaded because the
per-90 analysis depends on lineup position spans.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
COMPETITION, SEASON = 2, 44


def fetch_json(url: str, timeout: int = 30):
    req = Request(url, headers={"User-Agent": "football-xg-analytics/0.8"})
    with urlopen(req, timeout=timeout) as fh:
        content_type = fh.headers.get("Content-Type", "")
        raw = fh.read()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response from {url} is not valid JSON") from exc
    if "html" in content_type.lower():
        raise ValueError(f"response from {url} unexpectedly looks like HTML")
    return obj


def _validate_matches(matches) -> None:
    if not isinstance(matches, list) or not matches:
        raise ValueError("match manifest is empty or malformed")
    required = {"match_id", "match_date", "home_team", "away_team"}
    if any(not required.issubset(m) for m in matches):
        raise ValueError("match manifest is missing required fields")


def _validate_events(events, match_id: int) -> None:
    if not isinstance(events, list) or not events:
        raise ValueError(f"events for {match_id} are empty or malformed")
    if not all(isinstance(e, dict) and "type" in e for e in events[:20]):
        raise ValueError(f"events for {match_id} do not look like StatsBomb events")


def _validate_lineups(lineups, match_id: int) -> None:
    if not isinstance(lineups, list) or len(lineups) != 2:
        raise ValueError(f"lineups for {match_id} are malformed")
    if not all("team_name" in side and "lineup" in side for side in lineups):
        raise ValueError(f"lineups for {match_id} are missing expected fields")


def download(matches_limit: int | None = None, raw_dir: Path = RAW) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    matches = fetch_json(f"{BASE}/matches/{COMPETITION}/{SEASON}.json")
    _validate_matches(matches)
    matches = sorted(matches, key=lambda m: (m["match_date"], m["match_id"]))
    (raw_dir / "matches.json").write_text(json.dumps(matches, indent=2))
    selected = matches[:matches_limit] if matches_limit else matches
    (raw_dir / "matches_selected.json").write_text(json.dumps(selected, indent=2))

    for m in selected:
        mid = int(m["match_id"])
        event_dest = raw_dir / f"events_{mid}.json"
        lineup_dest = raw_dir / f"lineups_{mid}.json"
        if not event_dest.exists():
            events = fetch_json(f"{BASE}/events/{mid}.json")
            _validate_events(events, mid)
            event_dest.write_text(json.dumps(events))
        if not lineup_dest.exists():
            lineups = fetch_json(f"{BASE}/lineups/{mid}.json")
            _validate_lineups(lineups, mid)
            lineup_dest.write_text(json.dumps(lineups))
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=None, help="download the first N chronological Arsenal league matches")
    args = ap.parse_args()
    if args.matches is not None and args.matches <= 0:
        raise SystemExit("--matches must be positive")
    selected = download(args.matches)
    print(f"Downloaded/verified events + lineups for {len(selected)} matches -> {RAW}")


if __name__ == "__main__":
    main()
