"""Expected-goals modelling with grouped out-of-fold evaluation.

The central audit change in v0.8 is simple: headline model quality is never
computed on the same shots used to fit the coefficients.  Matches are the
cross-validation groups, so every reported probability is produced by a model
that did not see that match during fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - sklearn>=1.3 in requirements
    StratifiedGroupKFold = None
from sklearn.model_selection import GroupKFold

from shots import load_shots, shot_geometry, synthetic_shots

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
FEATURES = ["distance", "angle", "header", "set_piece"]
FEATURES_V2 = FEATURES + ["blockers", "gk_dist"]


def make_model() -> LogisticRegression:
    """Small, transparent probabilistic baseline used throughout."""
    return LogisticRegression(max_iter=2000)


def fit_xg(df: pd.DataFrame, features: Iterable[str] = FEATURES) -> LogisticRegression:
    model = make_model()
    model.fit(df[list(features)], df["goal"])
    return model


def _splitter(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    groups = df["match_id"].to_numpy()
    y = df["goal"].to_numpy()
    n_groups = int(pd.Series(groups).nunique())
    k = min(n_splits, n_groups)
    if k < 2:
        raise ValueError("at least two match groups are required for cross-fitting")
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(
            np.zeros(len(df)), y, groups
        )
    return GroupKFold(n_splits=k).split(np.zeros(len(df)), y, groups)


def cross_fitted_probabilities(
    df: pd.DataFrame,
    features: Iterable[str],
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Return grouped out-of-fold probabilities, holding whole matches out."""
    feats = list(features)
    required = set(feats + ["goal", "match_id"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns for cross-fitting: {sorted(missing)}")
    if df[feats].isna().any().any():
        raise ValueError("cross-fitting features contain missing values")
    p = np.full(len(df), np.nan, dtype=float)
    for train_idx, test_idx in _splitter(df, n_splits=n_splits, seed=seed):
        y_train = df.iloc[train_idx]["goal"]
        if y_train.nunique() < 2:
            raise ValueError("a training fold contains only one outcome class")
        model = fit_xg(df.iloc[train_idx], feats)
        p[test_idx] = model.predict_proba(df.iloc[test_idx][feats])[:, 1]
    if np.isnan(p).any():
        raise RuntimeError("cross-fitting did not produce a probability for every row")
    return p


def ece_score(y, p, bins: int = 8) -> float:
    """Equal-frequency expected calibration error."""
    xs, ys, ns = calibration_points(np.asarray(y), np.asarray(p), bins=bins)
    if not ns:
        return float("nan")
    w = np.asarray(ns, dtype=float) / np.sum(ns)
    return float(np.sum(w * np.abs(xs - ys)))


def probability_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    out = {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_8bin": float(ece_score(y, p, bins=8)),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    return out


def calibration_points(y, p, bins: int = 8):
    """Equal-frequency calibration bins with duplicate-edge protection."""
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(p)
    chunks = np.array_split(order, min(bins, len(order)))
    xs, ys, ns = [], [], []
    for idx in chunks:
        if len(idx) < 5:
            continue
        xs.append(float(p[idx].mean()))
        ys.append(float(y[idx].mean()))
        ns.append(int(len(idx)))
    return np.asarray(xs), np.asarray(ys), ns


def with_freeze_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Shots with both audited freeze-frame features available."""
    return df.dropna(subset=["blockers", "gk_dist"]).copy()


def fit_xgot(df: pd.DataFrame, use_oof: bool = True, seed: int = 42):
    """Post-shot model among on-target shots with known goalmouth placement.

    Returns the full-data model for inspection plus the filtered frame.  When
    possible, ``xgot_oof`` contains grouped out-of-fold probabilities used for
    any evaluation metric.
    """
    on = df[(df.on_target == 1) & df.end_y.notna() & df.end_z.notna()].copy()
    if len(on) < 60 or on.goal.nunique() < 2:
        return None, on
    on["wide"] = (on.end_y - 40.0).abs()
    on["high"] = on.end_z.clip(lower=0)
    feats = ["distance", "angle", "wide", "high"]
    model = fit_xg(on, feats)
    if use_oof and on.match_id.nunique() >= 5:
        on["xgot_oof"] = cross_fitted_probabilities(on, feats, n_splits=5, seed=seed)
    return model, on


def _rounded_metrics(metrics: dict) -> dict:
    return {k: (None if v is None else round(float(v), 4)) for k, v in metrics.items()}


def run_analysis(df: pd.DataFrame, label: str, seed: int = 42) -> dict:
    """Fit descriptive full-data coefficients and evaluate with match-group OOF."""
    full = fit_xg(df, FEATURES)
    df = df.copy()
    df["xg_oof"] = cross_fitted_probabilities(df, FEATURES, seed=seed)

    metrics = {
        "label": label,
        "shots": int(len(df)),
        "goals": int(df.goal.sum()),
        "evaluation": "5-fold stratified group cross-fitting by match_id",
        "geometry_oof": _rounded_metrics(probability_metrics(df.goal, df.xg_oof)),
        "full_fit_coefficients_descriptive_only": {
            f: round(float(c), 4) for f, c in zip(FEATURES, full.coef_[0])
        },
        "full_fit_intercept_descriptive_only": round(float(full.intercept_[0]), 4),
    }

    truth_col = None
    truth_label = None
    if df.statsbomb_xg.notna().any():
        truth_col = df.statsbomb_xg.to_numpy(dtype=float)
        truth_label = "oracle_probability" if label.startswith("synthetic") else "statsbomb_xg"
        metrics[f"geometry_corr_with_{truth_label}"] = round(
            float(np.corrcoef(df.xg_oof, truth_col)[0, 1]), 4
        )

    ff = with_freeze_frames(df)
    if len(ff) >= 100 and ff.goal.nunique() == 2 and ff.match_id.nunique() >= 5:
        p_geo = cross_fitted_probabilities(ff, FEATURES, seed=seed)
        p_ff = cross_fitted_probabilities(ff, FEATURES_V2, seed=seed)
        full_ff = fit_xg(ff, FEATURES_V2)
        ff_metrics = {
            "shots_with_frames": int(len(ff)),
            "geometry_same_rows_oof": _rounded_metrics(probability_metrics(ff.goal, p_geo)),
            "freeze_frame_oof": _rounded_metrics(probability_metrics(ff.goal, p_ff)),
            "delta_log_loss_freeze_minus_geometry": round(
                float(log_loss(ff.goal, p_ff) - log_loss(ff.goal, p_geo)), 5
            ),
            "delta_auc_freeze_minus_geometry": round(
                float(roc_auc_score(ff.goal, p_ff) - roc_auc_score(ff.goal, p_geo)), 5
            ),
            "full_fit_coef_blockers_descriptive_only": round(float(full_ff.coef_[0][4]), 4),
            "full_fit_coef_gk_dist_descriptive_only": round(float(full_ff.coef_[0][5]), 4),
        }
        if ff.statsbomb_xg.notna().any():
            t = ff.statsbomb_xg.to_numpy(dtype=float)
            ff_metrics[f"geometry_corr_with_{truth_label}"] = round(float(np.corrcoef(p_geo, t)[0, 1]), 4)
            ff_metrics[f"freeze_frame_corr_with_{truth_label}"] = round(float(np.corrcoef(p_ff, t)[0, 1]), 4)
        metrics["freeze_frame_comparison"] = ff_metrics

    # xA uses cross-fitted xG, so the shot does not help fit its own probability.
    if df.assist_player.notna().any():
        xa = (
            df.dropna(subset=["assist_player"])
            .groupby("assist_player")["xg_oof"]
            .sum()
            .sort_values(ascending=False)
        )
        metrics["top_xa_cross_fitted_xg"] = {
            k: round(float(v), 2) for k, v in xa.head(5).items()
        }

    xgot_model, on = fit_xgot(df, use_oof=True, seed=seed)
    if xgot_model is not None:
        block = {
            "on_target_shots": int(len(on)),
            "full_fit_coef_wide_descriptive_only": round(float(xgot_model.coef_[0][2]), 4),
            "full_fit_coef_high_descriptive_only": round(float(xgot_model.coef_[0][3]), 4),
        }
        if "xgot_oof" in on:
            block["oof"] = _rounded_metrics(probability_metrics(on.goal, on.xgot_oof))
        metrics["xgot"] = block

    return metrics, df, full


def make_figures(df: pd.DataFrame, full_model: LogisticRegression, label: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(5, 40, 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for header, style, name in [(0, "-", "foot"), (1, "--", "header")]:
        rows = pd.DataFrame({
            "distance": grid,
            "angle": [shot_geometry(120 - float(g), 40)[1] for g in grid],
            "header": header,
            "set_piece": 0,
        })
        ax.plot(grid, full_model.predict_proba(rows[FEATURES])[:, 1], style, lw=1.9, label=f"open play, {name}")
    ax.set(xlabel="Distance from goal centre (StatsBomb units)", ylabel="P(goal)", title=f"Descriptive fitted xG curves ({label})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "xg_curves.png", dpi=150); plt.close(fig)

    xs, ys, ns = calibration_points(df.goal.to_numpy(), df.xg_oof.to_numpy())
    fig2, ax2 = plt.subplots(figsize=(5.6, 5.2))
    upper = max(float(xs.max(initial=0.1)), float(ys.max(initial=0.1))) * 1.08
    ax2.plot([0, upper], [0, upper], "k--", lw=1, label="perfect calibration")
    ax2.plot(xs, ys, "o-", lw=1.6, label="grouped OOF model")
    for x_, y_, n_ in zip(xs, ys, ns):
        ax2.annotate(n_, (x_, y_), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax2.set(xlabel="Mean out-of-fold predicted xG", ylabel="Observed goal rate", title="Grouped out-of-fold calibration")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(FIGURES / "calibration.png", dpi=150); plt.close(fig2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.synthetic:
        df = synthetic_shots(seed=args.seed)
        label = "synthetic known-truth"
        out_name = "xg_model_synthetic.json"
    else:
        df = load_shots()
        label = "StatsBomb open data: Arsenal 2003/04 league matches"
        out_name = "xg_model.json"

    metrics, scored, full = run_analysis(df, label=label, seed=args.seed)
    make_figures(scored, full, label)
    out = ROOT / "reports" / out_name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Figures -> {FIGURES}")
    return scored


if __name__ == "__main__":
    main()
