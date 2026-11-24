"""Expected-threat (xT) value model with explicit possession-loss leakage.

The pitch is a finite-state Markov reward system.  A zone's value is the
probability a possession starting there eventually produces a goal.  Failed
passes stay in the move-attempt denominator, so possession losses reduce the
transition row sum instead of being silently renormalised away.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minutes import season_minutes

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "reports" / "figures"
NX, NY = 16, 12
PITCH_X, PITCH_Y = 120.0, 80.0


def zone_of(x: float, y: float) -> tuple[int, int]:
    if not (0 <= x <= PITCH_X and 0 <= y <= PITCH_Y):
        raise ValueError("coordinates must lie on the 120 x 80 pitch")
    ix = min(int(x / PITCH_X * NX), NX - 1)
    iy = min(int(y / PITCH_Y * NY), NY - 1)
    return ix, iy


def collect_actions(team: str = "Arsenal", raw: Path = RAW):
    """Collect successful moves, shot/goals and attempted-move denominators."""
    files = sorted(raw.glob("events_*.json"))
    if not files:
        raise FileNotFoundError(f"no StatsBomb event files in {raw}")
    moves, shots, goals = [], np.zeros((NX, NY)), np.zeros((NX, NY))
    starts = np.zeros((NX, NY))
    move_attempts = np.zeros((NX, NY))
    for path in files:
        events = json.loads(path.read_text())
        for e in events:
            if e.get("team", {}).get("name") != team:
                continue
            kind = e.get("type", {}).get("name")
            loc = e.get("location")
            if not loc:
                continue
            if kind == "Shot" and e.get("shot", {}).get("type", {}).get("name") != "Penalty":
                z = zone_of(*loc[:2])
                shots[z] += 1
                starts[z] += 1
                goals[z] += e["shot"]["outcome"]["name"] == "Goal"
            elif kind in ("Pass", "Carry"):
                key = kind.lower()
                end = e.get(key, {}).get("end_location")
                failed = (kind == "Pass" and "outcome" in e.get("pass", {})) or end is None
                z0 = zone_of(*loc[:2])
                starts[z0] += 1
                move_attempts[z0] += 1
                if failed:
                    continue
                z1 = zone_of(*end[:2])
                moves.append((z0, z1, e.get("player", {}).get("name")))
    return moves, shots, goals, starts, move_attempts


def xt_components(moves, shots, goals, starts, move_attempts=None):
    """Build immediate-goal reward and leaky movement transition matrix."""
    shots = np.asarray(shots, dtype=float)
    goals = np.asarray(goals, dtype=float)
    starts = np.asarray(starts, dtype=float)
    if shots.shape != (NX, NY) or goals.shape != (NX, NY) or starts.shape != (NX, NY):
        raise ValueError("shot/goal/start arrays must match the current xT grid")
    if np.any(goals > shots) or np.any(shots > starts):
        raise ValueError("counts violate goals <= shots <= starts")
    shoot_p = np.divide(shots, starts, out=np.zeros_like(shots), where=starts > 0)
    goal_p = np.divide(goals, shots, out=np.zeros_like(goals), where=shots > 0)
    move_p = 1.0 - shoot_p

    transition_cond = np.zeros((NX, NY, NX, NY))
    completed = np.zeros((NX, NY))
    for z0, z1, _ in moves:
        transition_cond[z0 + z1] += 1
        completed[z0] += 1
    denom = completed if move_attempts is None else np.asarray(move_attempts, dtype=float)
    if denom.shape != (NX, NY):
        raise ValueError("move_attempts must match the xT grid")
    if np.any(completed - denom > 1e-12):
        raise ValueError("completed moves cannot exceed attempted moves")
    transition_cond = np.divide(
        transition_cond,
        denom[:, :, None, None],
        out=np.zeros_like(transition_cond),
        where=denom[:, :, None, None] > 0,
    )
    # transition_cond row sum is move completion probability. Multiplying by
    # move_p turns it into the unconditional probability of a successful move
    # to each destination from an action opportunity in the zone.
    q = move_p[:, :, None, None] * transition_cond
    reward = shoot_p * goal_p
    if np.nanmax(q.reshape(NX * NY, -1).sum(axis=1), initial=0) > 1 + 1e-10:
        raise ValueError("xT transition probabilities exceed one")
    return reward, q


def fit_xt(moves, shots, goals, starts, move_attempts=None, max_iters: int = 500, tol: float = 1e-12, return_diagnostics: bool = False):
    """Value iteration for xT, stopping by numerical convergence."""
    reward, q = xt_components(moves, shots, goals, starts, move_attempts)
    xt = np.zeros((NX, NY))
    delta = float("inf")
    for i in range(1, max_iters + 1):
        new = reward + np.einsum("abcd,cd->ab", q, xt)
        delta = float(np.max(np.abs(new - xt)))
        xt = new
        if delta < tol:
            break
    diagnostics = {"iterations": i, "max_update": delta, "converged": delta < tol}
    if not diagnostics["converged"]:
        raise RuntimeError("xT value iteration did not converge")
    return (xt, diagnostics) if return_diagnostics else xt


def solve_xt_exact(moves, shots, goals, starts, move_attempts=None) -> np.ndarray:
    """Independent linear-system solution: (I-Q)V = reward."""
    reward, q = xt_components(moves, shots, goals, starts, move_attempts)
    n = NX * NY
    q2 = q.reshape(n, n)
    a = np.eye(n) - q2
    return np.linalg.solve(a, reward.reshape(n)).reshape(NX, NY)


def player_xt_added(moves, xt) -> pd.Series:
    gained: dict[str, float] = {}
    for z0, z1, player in moves:
        if player:
            gained[player] = gained.get(player, 0.0) + float(xt[z1] - xt[z0])
    return pd.Series(gained, name="xt_added").sort_values(ascending=False)


def main():
    moves, shots, goals, starts, attempts = collect_actions()
    xt, diag = fit_xt(moves, shots, goals, starts, attempts, return_diagnostics=True)
    exact = solve_xt_exact(moves, shots, goals, starts, attempts)
    parity = float(np.max(np.abs(xt - exact)))

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    im = ax.imshow(xt.T, origin="lower", extent=[0, PITCH_X, 0, PITCH_Y], aspect="auto")
    fig.colorbar(im, ax=ax, label="P(possession here eventually produces a goal)")
    ax.set(title=f"Expected threat surface, Arsenal 2003/04 ({len(moves):,} successful moves)", xlabel="toward opposition goal")
    fig.tight_layout(); fig.savefig(FIGURES / "xt_surface.png", dpi=150); plt.close(fig)

    added = player_xt_added(moves, xt)
    mins = season_minutes()
    table = pd.DataFrame({"xt_added": added}).join(mins, how="inner")
    table = table[table.minutes >= 900].copy()
    table["xt_per90"] = table.xt_added / table.minutes * 90
    table = table.sort_values("xt_per90", ascending=False)

    summary = {
        "moves_counted": len(moves),
        "max_zone_value": round(float(xt.max()), 4),
        "value_iteration": diag,
        "max_abs_difference_vs_exact_linear_solve": parity,
        "top_xt_per90": {k: round(float(v), 3) for k, v in table.xt_per90.head(8).items()},
    }
    (ROOT / "reports" / "xt_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
