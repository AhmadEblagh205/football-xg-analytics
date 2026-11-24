"""Player finishing and information-priority analysis.

v0.8 replaces the earlier ad-hoc goals/xG shrinkage variance with a coherent
regularised log-odds-offset model.  For player j,

    logit P(goal_i) = logit(xG_i) + alpha_j,
    alpha_j ~ N(0, prior_sd^2).

A Laplace approximation gives posterior variance for alpha_j.  The optional
"learning priority" is the expected reduction in that variance from a typical
match's worth of future shots; it is an information diagnostic, not a claim
about transfer value or whom a club should sign.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from minutes import season_minutes
from model import FEATURES, cross_fitted_probabilities
from shots import load_shots, synthetic_shots

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def _player_map(y: np.ndarray, p: np.ndarray, prior_sd: float) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    y = np.asarray(y, dtype=float)
    base = logit(p)

    def objective(alpha: float) -> float:
        q = np.clip(expit(base + alpha), 1e-10, 1 - 1e-10)
        ll = np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
        return float(-ll + 0.5 * (alpha / prior_sd) ** 2)

    res = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not res.success:
        raise RuntimeError("player finishing MAP optimisation failed")
    alpha = float(res.x)
    q = expit(base + alpha)
    precision = 1.0 / prior_sd**2 + float(np.sum(q * (1 - q)))
    return alpha, 1.0 / precision


def player_table(
    df: pd.DataFrame,
    min_shots: int = 8,
    prior_sd: float = 0.75,
) -> pd.DataFrame:
    """Return player volume, xG and regularised finishing-offset estimates."""
    if "xg" not in df:
        raise ValueError("df must contain an xg probability column")
    if prior_sd <= 0:
        raise ValueError("prior_sd must be positive")
    n_matches = max(int(df.match_id.nunique()), 1)
    rows = []
    for player, d in df.groupby("player"):
        if len(d) < min_shots:
            continue
        alpha, var = _player_map(d.goal.to_numpy(), d.xg.to_numpy(), prior_sd)
        mean_xg = float(d.xg.mean())
        q_typical = float(expit(logit(np.clip(mean_xg, 1e-5, 1 - 1e-5)) + alpha))
        info_per_shot = q_typical * (1 - q_typical)
        shots_per_match = len(d) / n_matches
        info_per_match = info_per_shot * shots_per_match
        # Laplace precision update: new variance = 1 / (1/V + I).
        variance_reduction = var - 1.0 / (1.0 / var + info_per_match)
        rows.append({
            "player": player,
            "shots": int(len(d)),
            "goals": int(d.goal.sum()),
            "xg": float(d.xg.sum()),
            "goals_minus_xg": float(d.goal.sum() - d.xg.sum()),
            "mean_xg": mean_xg,
            "finishing_logodds_offset": alpha,
            "odds_multiplier": float(np.exp(alpha)),
            "offset_se_laplace": float(np.sqrt(var)),
            "offset_ci_low": float(alpha - 1.96 * np.sqrt(var)),
            "offset_ci_high": float(alpha + 1.96 * np.sqrt(var)),
            "info_per_shot": info_per_shot,
            "shots_per_match": shots_per_match,
            "expected_var_reduction_per_match": variance_reduction,
        })
    table = pd.DataFrame(rows).set_index("player") if rows else pd.DataFrame()
    if table.empty:
        return table
    m = table.expected_var_reduction_per_match.max()
    table["learning_priority"] = table.expected_var_reduction_per_match / m if m > 0 else 0.0
    return table.sort_values("xg", ascending=False)


def ranking_figure(table: pd.DataFrame, col: str, title: str, path: Path, top: int = 12):
    if table.empty:
        return
    t = table.sort_values(col).tail(top)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.6))
    ax.hlines(range(len(t)), 0, t[col], lw=2)
    ax.plot(t[col], range(len(t)), "o")
    ax.set_yticks(range(len(t)), t.index, fontsize=8)
    ax.set(title=title)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--min-shots", type=int, default=8)
    ap.add_argument("--min-minutes", type=float, default=900.0)
    ap.add_argument("--prior-sd", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = synthetic_shots(3000, seed=args.seed) if args.synthetic else load_shots()
    # Cross-fitted xG prevents each shot from helping fit its own baseline probability.
    df = df.copy()
    df["xg"] = cross_fitted_probabilities(df, FEATURES, seed=args.seed)
    table = player_table(df, min_shots=args.min_shots, prior_sd=args.prior_sd)

    if not args.synthetic:
        mins = season_minutes()
        if not mins.empty:
            table = table.join(mins, how="left")
            enough = table.minutes >= args.min_minutes
            for col in ["xg", "goals", "shots"]:
                table.loc[enough, f"{col}_per90"] = table.loc[enough, col] / table.loc[enough, "minutes"] * 90

    FIGURES.mkdir(parents=True, exist_ok=True)
    ranking_figure(table, "xg", "Cross-fitted non-penalty xG", FIGURES / "scout_xg.png")
    ranking_figure(table, "finishing_logodds_offset", "Regularised finishing log-odds offset", FIGURES / "scout_finishing.png")
    ranking_figure(table, "learning_priority", "Expected finishing-information gain per typical match", FIGURES / "scout_priority.png")
    if "xg_per90" in table:
        ranking_figure(table.dropna(subset=["xg_per90"]), "xg_per90", f"Cross-fitted non-penalty xG per 90 (min {args.min_minutes:g} min)", FIGURES / "scout_xg_per90.png")

    cols = [
        "shots", "goals", "xg", "goals_minus_xg", "finishing_logodds_offset",
        "odds_multiplier", "offset_se_laplace", "offset_ci_low", "offset_ci_high",
        "learning_priority",
    ]
    if "xg_per90" in table:
        cols += ["minutes", "xg_per90", "goals_per90", "shots_per90"]
    report = {
        "label": "synthetic known-truth baseline" if args.synthetic else "StatsBomb Arsenal 2003/04 open data",
        "method": "player log-odds finishing offset with N(0, prior_sd^2) regularisation and Laplace uncertainty",
        "prior_sd": args.prior_sd,
        "players": json.loads(table[cols].round(4).to_json(orient="index")),
    }
    out = ROOT / "reports" / ("scouting_synthetic.json" if args.synthetic else "scouting.json")
    out.write_text(json.dumps(report, indent=2))
    print(table[cols].round(3).head(12).to_string())
    print(f"\nSaved -> {out}\nFigures -> {FIGURES}")


if __name__ == "__main__":
    main()
