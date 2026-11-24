"""Minutes played from the StatsBomb lineup files. Each player's
positions list carries from and to clocks with their periods; summing the
spans (an open to means played to the end of that period) gives minutes,
and minutes are what turn season totals into per 90 rates, the only
scale on which a starter and a super-sub can be compared.
"""
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PERIOD_END = {1: 45.0, 2: 90.0}


def _clock_to_min(clock: str) -> float:
    mm, ss = clock.split(":")[:2]
    return int(mm) + int(ss) / 60


def player_minutes(player: dict) -> float:
    total = 0.0
    for pos in player.get("positions", []):
        start = _clock_to_min(pos["from"]) if pos.get("from") else 0.0
        if pos.get("to"):
            end = _clock_to_min(pos["to"])
        else:
            # an open "to" means played to the end: of that period if one
            # is stated, otherwise of the match
            end = PERIOD_END.get(pos.get("to_period"), 90.0)
        total += max(end - start, 0.0)
    return min(total, 98.0)


def season_minutes(team: str = "Arsenal") -> pd.Series:
    """Total minutes per player for one team across every lineup file."""
    totals: dict[str, float] = {}
    for path in sorted(RAW.glob("lineups_*.json")):
        for side in json.loads(path.read_text()):
            if side["team_name"] != team:
                continue
            for player in side["lineup"]:
                name = player["player_name"]
                totals[name] = totals.get(name, 0.0) + player_minutes(player)
    return pd.Series(totals, name="minutes").sort_values(ascending=False)
