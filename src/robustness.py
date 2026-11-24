"""Known-truth robustness experiments for the xG feature comparison.

The experiment asks whether adding blocker and keeper-distance features earns
its extra complexity when those features are (a) truly irrelevant or (b)
actually present in the data-generating process.  All model comparisons use
match-group out-of-fold predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from model import FEATURES, FEATURES_V2, cross_fitted_probabilities
from shots import synthetic_shots

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def one_world(seed: int, n: int, freeze_strength: float) -> dict:
    df = synthetic_shots(n=n, seed=seed, freeze_strength=freeze_strength)
    p0 = cross_fitted_probabilities(df, FEATURES, seed=seed)
    p1 = cross_fitted_probabilities(df, FEATURES_V2, seed=seed)
    truth = df.statsbomb_xg.to_numpy(float)
    return {
        "seed": seed,
        "n": n,
        "freeze_strength": freeze_strength,
        "auc_geometry": roc_auc_score(df.goal, p0),
        "auc_freeze": roc_auc_score(df.goal, p1),
        "delta_auc": roc_auc_score(df.goal, p1) - roc_auc_score(df.goal, p0),
        "logloss_geometry": log_loss(df.goal, p0),
        "logloss_freeze": log_loss(df.goal, p1),
        "delta_logloss": log_loss(df.goal, p1) - log_loss(df.goal, p0),
        "brier_geometry": brier_score_loss(df.goal, p0),
        "brier_freeze": brier_score_loss(df.goal, p1),
        "delta_brier": brier_score_loss(df.goal, p1) - brier_score_loss(df.goal, p0),
        "corr_geometry_oracle": np.corrcoef(p0, truth)[0, 1],
        "corr_freeze_oracle": np.corrcoef(p1, truth)[0, 1],
    }


def run(worlds: int = 20) -> pd.DataFrame:
    rows = []
    for n in (450, 900, 1800):
        for strength in (0.0, 1.0):
            for seed in range(worlds):
                rows.append(one_world(1000 * n + 100 * int(strength) + seed, n, strength))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    rows = []
    for (n, strength), g in df.groupby(["n", "freeze_strength"], sort=True):
        rows.append({
            "shots": int(n),
            "freeze_signal": "absent" if strength == 0 else "planted",
            "worlds": int(len(g)),
            "median_delta_auc": round(float(g.delta_auc.median()), 4),
            "median_delta_logloss": round(float(g.delta_logloss.median()), 5),
            "median_delta_brier": round(float(g.delta_brier.median()), 5),
            "freeze_better_logloss_fraction": round(float((g.delta_logloss < 0).mean()), 3),
            "freeze_better_auc_fraction": round(float((g.delta_auc > 0).mean()), 3),
            "median_geometry_oracle_corr": round(float(g.corr_geometry_oracle.median()), 4),
            "median_freeze_oracle_corr": round(float(g.corr_freeze_oracle.median()), 4),
        })
    return {"design": "known-truth synthetic worlds; 5-fold stratified group CV by match", "conditions": rows}


def plot(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    positions, labels, data = [], [], []
    pos = 1
    for n in (450, 900, 1800):
        for strength in (0.0, 1.0):
            g = df[(df.n == n) & (df.freeze_strength == strength)]
            data.append((-g.delta_logloss).to_numpy())  # positive = freeze model better
            positions.append(pos)
            labels.append(f"{n}\n{'signal' if strength else 'none'}")
            pos += 1
        pos += 0.4
    ax.boxplot(data, positions=positions, widths=0.65, showfliers=False)
    ax.axhline(0, lw=1, ls="--")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Held-out log-loss improvement from freeze-frame features")
    ax.set_xlabel("Shots and planted freeze-frame signal")
    ax.set_title("Extra features earn their place only when the signal exists")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(FIGURES / "freeze_frame_robustness.png", dpi=150); plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.4, 4.6))
    for strength, label in [(0.0, "no planted signal"), (1.0, "planted signal")]:
        med = (df[df.freeze_strength == strength]
               .groupby("n").delta_auc.median())
        ax2.plot(med.index, med.values, marker="o", label=label)
    ax2.axhline(0, lw=1, ls="--")
    ax2.set(xlabel="Number of shots", ylabel="Median OOF AUC change",
            title="Sample size and functional form both matter")
    ax2.legend(); ax2.grid(alpha=0.25)
    fig2.tight_layout(); fig2.savefig(FIGURES / "freeze_frame_sample_size.png", dpi=150); plt.close(fig2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=20)
    args = ap.parse_args()
    df = run(args.worlds)
    summary = summarise(df)
    ROOT.joinpath("reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports" / "freeze_frame_worlds.csv", index=False)
    (ROOT / "reports" / "robustness.json").write_text(json.dumps(summary, indent=2))
    plot(df)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
