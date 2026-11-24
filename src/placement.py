"""Post-shot placement evaluation with match-group out-of-fold scoring.

The real-data result is descriptive and must be regenerated after downloading
StatsBomb data.  CI and the checked-in v0.8 audit use a known-truth synthetic
negative control where goalmouth placement is random but keeper distance is
part of the generating model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from model import cross_fitted_probabilities, probability_metrics, fit_xg
from shots import load_shots, synthetic_shots

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
GOAL_LO, GOAL_HI, BAR = 36.0, 44.0, 2.67
PRE = ["distance", "angle", "header"]
PLACED = PRE + ["dy", "end_z", "corner"]
KEEPER = PRE + ["gk_dist"]


def placement_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dy"] = (d.end_y - 40.0).abs().clip(0, 4.0)
    d["end_z"] = d.end_z.clip(0, BAR)
    # Normalised Euclidean distance from the nearest upper corner.
    d["corner"] = np.sqrt(((4.0 - d.dy) / 4.0) ** 2 + ((BAR - d.end_z) / BAR) ** 2)
    return d


def fit_models(d: pd.DataFrame):
    return fit_xg(d, PRE), fit_xg(d, PLACED), fit_xg(d, KEEPER)


def evaluate_models(d: pd.DataFrame, seed: int = 42) -> dict:
    p_pre = cross_fitted_probabilities(d, PRE, seed=seed)
    p_placed = cross_fitted_probabilities(d, PLACED, seed=seed)
    p_keeper = cross_fitted_probabilities(d, KEEPER, seed=seed)
    return {
        "on_target_shots": int(len(d)),
        "evaluation": "5-fold stratified group cross-fitting by match_id",
        "pre_strike_geometry_oof": {k: round(v, 4) if v is not None else None for k, v in probability_metrics(d.goal, p_pre).items()},
        "with_placement_oof": {k: round(v, 4) if v is not None else None for k, v in probability_metrics(d.goal, p_placed).items()},
        "with_keeper_distance_oof": {k: round(v, 4) if v is not None else None for k, v in probability_metrics(d.goal, p_keeper).items()},
        "delta_auc_placement_minus_pre": round(float(roc_auc_score(d.goal, p_placed) - roc_auc_score(d.goal, p_pre)), 5),
        "delta_auc_keeper_minus_pre": round(float(roc_auc_score(d.goal, p_keeper) - roc_auc_score(d.goal, p_pre)), 5),
    }


def keeper_view(placed, d: pd.DataFrame, path: Path):
    ny, nz = 60, 30
    ys = np.linspace(GOAL_LO, GOAL_HI, ny)
    zs = np.linspace(0, BAR, nz)
    base = d[PRE].median()
    rows = []
    for z in zs:
        for y in ys:
            dy = min(abs(y - 40.0), 4.0)
            corner = np.sqrt(((4.0 - dy) / 4.0) ** 2 + ((BAR - z) / BAR) ** 2)
            rows.append([*base.to_numpy(), dy, z, corner])
    grid = placed.predict_proba(pd.DataFrame(rows, columns=PLACED))[:, 1]
    surface = grid.reshape(nz, ny)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    im = ax.imshow(surface, origin="lower", extent=[GOAL_LO, GOAL_HI, 0, BAR], aspect="auto")
    fig.colorbar(im, ax=ax, label="Fitted P(goal | on target)")
    ax.set(xlabel="across the goal mouth", ylabel="height", title="Descriptive placement surface")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = synthetic_shots(3000, seed=args.seed, freeze_strength=1.0) if args.synthetic else load_shots()
    d = placement_features(raw.dropna(subset=["end_y", "end_z", "gk_dist"]))
    d = d[d.on_target == 1].copy()
    if len(d) < 100 or d.goal.nunique() < 2:
        raise ValueError("not enough on-target shots with both outcome classes")
    summary = evaluate_models(d, seed=args.seed)
    summary["label"] = "synthetic negative-control placement" if args.synthetic else "StatsBomb Arsenal 2003/04 open data"

    _, placed, _ = fit_models(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    keeper_view(placed, d, FIGURES / "keeper_view.png")

    out = ROOT / "reports" / ("placement_synthetic.json" if args.synthetic else "placement_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
