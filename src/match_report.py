"""One-command match report using leave-one-match-out xG probabilities.

The selected match is never used to fit the model that scores its shots.  This
keeps the report descriptive but prevents the match from influencing its own
xG estimates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from model import FEATURES, fit_xg
from shots import load_shots

ROOT = Path(__file__).resolve().parents[1]


def match_frame(match_id: int, df=None):
    df = load_shots() if df is None else df.copy()
    target = df[df.match_id == match_id].copy()
    train = df[df.match_id != match_id]
    if target.empty:
        raise ValueError(f"No shots found for match {match_id}")
    if train.empty or train.goal.nunique() < 2:
        raise ValueError("not enough other matches to fit leave-one-match-out xG")
    model = fit_xg(train, FEATURES)
    target["xg"] = model.predict_proba(target[FEATURES])[:, 1]
    return target.sort_values(["minute", "player"])


def xg_race(m, out: Path):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    max_minute = max(95, int(m.minute.max()) + 1)
    for team, g in m.groupby("team"):
        minutes = [0] + g.minute.tolist() + [max_minute]
        cum = [0.0] + g.xg.cumsum().tolist()
        cum = cum + [cum[-1]]
        ax.step(minutes, cum, where="post", lw=2, label=f"{team} ({int(g.goal.sum())} goals, {g.xg.sum():.2f} xG)")
        for minute in g.loc[g.goal == 1, "minute"]:
            ax.axvline(minute, lw=0.6, alpha=0.45)
    ax.set(xlabel="Minute", ylabel="Cumulative leave-one-match-out xG", title="xG race")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def shot_map(m, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for team, g in m.groupby("team"):
        ax.scatter(g.x, g.y, s=600 * g.xg + 15, alpha=0.55, label=team)
        goals = g[g.goal == 1]
        ax.scatter(goals.x, goals.y, s=600 * goals.xg + 60, facecolors="none", edgecolors="black", linewidths=1.6)
    ax.set(xlim=(60, 121), ylim=(80, 0), title="Shot map, size = leave-one-match-out xG, ringed = goal")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def build_summary(m):
    summary = {}
    for team, g in m.groupby("team"):
        summary[team] = {"goals": int(g.goal.sum()), "xg": round(float(g.xg.sum()), 2), "shots": int(len(g))}
    leaders = sorted(summary, key=lambda t: summary[t]["xg"], reverse=True)
    if len(leaders) >= 2:
        summary["verdict"] = f"{leaders[0]} created more leave-one-match-out xG ({summary[leaders[0]]['xg']} vs {summary[leaders[1]]['xg']})"
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    df = load_shots()
    if args.list or not args.match:
        for mid, g in df.groupby("match_id"):
            teams = " vs ".join(sorted(g.team.unique()))
            score = g.groupby("team").goal.sum().astype(int).to_dict()
            print(f"{mid}: {teams}  {score}")
        return
    m = match_frame(args.match, df=df)
    out_dir = ROOT / "reports" / f"match_{args.match}"
    out_dir.mkdir(parents=True, exist_ok=True)
    xg_race(m, out_dir / "xg_race.png")
    shot_map(m, out_dir / "shot_map.png")
    summary = build_summary(m)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
