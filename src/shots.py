"""Shot-table construction and controlled synthetic data.

StatsBomb coordinates use a 120 x 80 pitch with the attacked goal at
x=120 and posts at y=36 and y=44.  The real-data parser keeps provenance
explicit; the synthetic generator is a known-truth laboratory used by CI
and robustness experiments.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
GOAL_X, POST_LOW, POST_HIGH = 120.0, 36.0, 44.0
GOAL_CENTRE = (GOAL_X, 40.0)
SET_PIECE_PATTERNS = {"From Corner", "From Free Kick", "From Throw In"}


def _sign(p, a, b):
    return (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])


def in_shot_cone(point, shot_xy) -> bool:
    """Return whether a point lies in the triangle from shot to both posts."""
    a, b, c = shot_xy, (GOAL_X, POST_LOW), (GOAL_X, POST_HIGH)
    d1, d2, d3 = _sign(point, a, b), _sign(point, b, c), _sign(point, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def freeze_frame_features(shot_event) -> tuple[float, float]:
    """Return blocker count and opposition-keeper distance from goal centre.

    Missing freeze frames remain missing rather than being silently imputed.
    """
    frame = shot_event["shot"].get("freeze_frame")
    x, y = shot_event["location"][0], shot_event["location"][1]
    if not frame:
        return np.nan, np.nan
    blockers, gk_dist = 0, np.nan
    for player in frame:
        px, py = player["location"][0], player["location"][1]
        is_gk = player.get("position", {}).get("name") == "Goalkeeper"
        if not player.get("teammate", False) and is_gk:
            gk_dist = math.hypot(GOAL_CENTRE[0] - px, GOAL_CENTRE[1] - py)
        elif not player.get("teammate", False):
            blockers += int(in_shot_cone((px, py), (x, y)))
    return float(blockers), float(gk_dist)


def shot_geometry(x: float, y: float) -> tuple[float, float]:
    """Distance to goal centre and angle subtended by the goal mouth."""
    if not (0 <= x <= GOAL_X and 0 <= y <= 80):
        raise ValueError("shot coordinates must lie on the StatsBomb 120 x 80 pitch")
    dist = math.hypot(GOAL_X - x, 40.0 - y)
    a = math.hypot(GOAL_X - x, POST_LOW - y)
    b = math.hypot(GOAL_X - x, POST_HIGH - y)
    width = POST_HIGH - POST_LOW
    cos_g = (a * a + b * b - width * width) / (2 * a * b) if a * b else 1.0
    angle = math.acos(max(-1.0, min(1.0, cos_g)))
    return dist, angle


def _validate_event_file(events, path: Path) -> None:
    if not isinstance(events, list):
        raise ValueError(f"{path} is not a StatsBomb event list")
    if events and not all(isinstance(e, dict) and "type" in e for e in events[:20]):
        raise ValueError(f"{path} does not look like StatsBomb event data")


def load_shots(raw: Path = RAW) -> pd.DataFrame:
    """Build one row per non-penalty shot across downloaded event files."""
    rows = []
    files = sorted(raw.glob("events_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No event files in {raw}; run src/download_data.py or use synthetic_shots()"
        )
    for f in files:
        events = json.loads(f.read_text())
        _validate_event_file(events, f)
        passes = {
            e["id"]: e for e in events
            if e.get("type", {}).get("name") == "Pass" and "id" in e
        }
        for e in events:
            if e.get("type", {}).get("name") != "Shot":
                continue
            s = e.get("shot", {})
            if s.get("type", {}).get("name") == "Penalty":
                continue
            if "location" not in e or len(e["location"]) < 2:
                continue
            x, y = float(e["location"][0]), float(e["location"][1])
            dist, angle = shot_geometry(x, y)
            end = s.get("end_location", [None, None, None])
            key = s.get("key_pass_id")
            blockers, gk_dist = freeze_frame_features(e)
            outcome = s.get("outcome", {}).get("name")
            rows.append({
                "match_id": int(f.stem.split("_")[1]),
                "player": e.get("player", {}).get("name", "Unknown"),
                "team": e.get("team", {}).get("name", "Unknown"),
                "x": x,
                "y": y,
                "distance": dist,
                "angle": angle,
                "header": int(s.get("body_part", {}).get("name") == "Head"),
                "minute": int(e.get("minute", 0)),
                "blockers": blockers,
                "gk_dist": gk_dist,
                "set_piece": int(e.get("play_pattern", {}).get("name") in SET_PIECE_PATTERNS),
                "on_target": int(outcome in ("Goal", "Saved", "Saved To Post")),
                "end_y": end[1] if len(end) > 1 else None,
                "end_z": end[2] if len(end) > 2 else None,
                "goal": int(outcome == "Goal"),
                "statsbomb_xg": s.get("statsbomb_xg"),
                "assist_player": passes[key].get("player", {}).get("name") if key in passes else None,
            })
    if not rows:
        raise ValueError("Downloaded event files contain no non-penalty shots")
    return pd.DataFrame(rows)


def synthetic_shots(
    n: int = 1200,
    seed: int = 7,
    freeze_strength: float = 1.0,
    n_matches: int = 38,
) -> pd.DataFrame:
    """Generate a known-truth shot sample with match groups.

    ``freeze_strength=0`` creates a world where blocker/keeper features carry
    no extra goal information beyond geometry. ``freeze_strength=1`` plants
    their default effect. ``statsbomb_xg`` stores the exact generating goal
    probability and is therefore an oracle target, not an external model.
    """
    if n < 100:
        raise ValueError("n must be at least 100 for a stable synthetic sample")
    if n_matches < 5:
        raise ValueError("n_matches must be at least 5")
    rng = np.random.default_rng(seed)
    x = np.clip(120 - rng.gamma(2.0, 6.0, n), 60, 119.5)
    y = np.clip(rng.normal(40, 12, n), 2, 78)
    geo = np.array([shot_geometry(float(xi), float(yi)) for xi, yi in zip(x, y)])
    dist, angle = geo[:, 0], geo[:, 1]
    header = rng.binomial(1, 0.2, n)
    set_piece = rng.binomial(1, 0.20, n)
    blockers = rng.poisson(np.clip(2.2 - 0.06 * dist + 3 * angle, 0.1, 6))
    gk_dist = np.clip(rng.normal(2.0, 1.0, n), 0.3, 8.0)

    base_logit = 0.8 - 0.10 * dist + 1.8 * angle - 0.9 * header - 0.3 * set_piece
    freeze_term = freeze_strength * (-0.35 * blockers + 0.10 * gk_dist)
    logit = base_logit + freeze_term
    p = 1.0 / (1.0 + np.exp(-logit))
    goal = rng.binomial(1, p)

    # On-target status is consistent with goals. Placement is intentionally
    # uninformative by default so it can serve as a negative-control feature.
    p_on_non_goal = np.clip(0.20 + 1.6 * p, 0.10, 0.85)
    on_target = np.where(goal == 1, 1, rng.binomial(1, p_on_non_goal))
    end_y = np.where(on_target == 1, rng.uniform(36.1, 43.9, n), np.nan)
    end_z = np.where(on_target == 1, rng.uniform(0.05, 2.55, n), np.nan)

    players = rng.choice([f"Player {c}" for c in "ABCDEFGHIJ"], n)
    match_ids = rng.integers(1, n_matches + 1, n)
    return pd.DataFrame({
        "match_id": match_ids,
        "player": players,
        "team": rng.choice(["Reds", "Blues"], n),
        "x": x,
        "y": y,
        "distance": dist,
        "angle": angle,
        "header": header,
        "set_piece": set_piece,
        "minute": rng.integers(1, 95, n),
        "blockers": blockers.astype(float),
        "gk_dist": gk_dist,
        "on_target": on_target.astype(int),
        "end_y": end_y,
        "end_z": end_z,
        "goal": goal,
        "statsbomb_xg": p,
        "assist_player": None,
    })
