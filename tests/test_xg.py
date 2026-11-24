import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from model import FEATURES, FEATURES_V2, calibration_points, cross_fitted_probabilities, fit_xg
from shots import SET_PIECE_PATTERNS, shot_geometry, synthetic_shots


def test_geometry_behaves():
    d_close, a_close = shot_geometry(114, 40)
    d_far, a_far = shot_geometry(90, 40)
    assert d_close < d_far
    assert a_close > a_far
    _, a_wide = shot_geometry(114, 10)
    assert a_wide < a_close


def test_geometry_rejects_off_pitch_coordinates():
    import pytest
    with pytest.raises(ValueError):
        shot_geometry(121, 40)


def test_set_piece_definition_is_explicit_not_every_non_regular_pattern():
    assert "From Corner" in SET_PIECE_PATTERNS
    assert "From Free Kick" in SET_PIECE_PATTERNS
    assert "From Goal Kick" not in SET_PIECE_PATTERNS
    assert "From Keeper" not in SET_PIECE_PATTERNS


def test_xg_recovers_signal_direction():
    df = synthetic_shots(5000, seed=1)
    m = fit_xg(df)
    coef = dict(zip(FEATURES, m.coef_[0]))
    assert coef["distance"] < 0
    assert coef["angle"] > 0
    assert coef["header"] < 0


def test_group_crossfit_returns_valid_probabilities():
    df = synthetic_shots(1200, seed=2)
    p = cross_fitted_probabilities(df, FEATURES, seed=2)
    assert len(p) == len(df)
    assert np.all((p > 0) & (p < 1))


def test_freeze_features_help_when_signal_is_planted():
    df = synthetic_shots(2500, seed=3, freeze_strength=1.0)
    p0 = cross_fitted_probabilities(df, FEATURES, seed=3)
    p1 = cross_fitted_probabilities(df, FEATURES_V2, seed=3)
    assert log_loss(df.goal, p1) < log_loss(df.goal, p0) - 0.01


def test_freeze_features_do_not_create_large_gain_when_signal_absent():
    df = synthetic_shots(2500, seed=4, freeze_strength=0.0)
    p0 = cross_fitted_probabilities(df, FEATURES, seed=4)
    p1 = cross_fitted_probabilities(df, FEATURES_V2, seed=4)
    # Extra irrelevant features may wiggle by sampling noise but should not
    # manufacture a material held-out improvement.
    assert log_loss(df.goal, p1) > log_loss(df.goal, p0) - 0.005


def test_calibration_bins_are_sane_on_oof_predictions():
    df = synthetic_shots(1500, seed=5)
    p = cross_fitted_probabilities(df, FEATURES, seed=5)
    xs, ys, ns = calibration_points(df.goal.to_numpy(), p)
    assert len(xs) >= 5
    assert sum(ns) == len(df)
    assert np.all((ys >= 0) & (ys <= 1))


def test_shot_cone_membership():
    from shots import in_shot_cone
    shot = (108.0, 40.0)
    assert in_shot_cone((114.0, 40.0), shot)
    assert not in_shot_cone((114.0, 20.0), shot)
    assert not in_shot_cone((100.0, 40.0), shot)


def test_synthetic_goals_are_always_on_target():
    df = synthetic_shots(2000, seed=6)
    assert (df.loc[df.goal == 1, "on_target"] == 1).all()


def test_minutes_parser_handles_subs_and_open_ends():
    from minutes import player_minutes
    starter = {"positions": [{"from": "00:00", "to": None, "from_period": 1, "to_period": 1}]}
    assert player_minutes(starter) == 45.0
    full = {"positions": [
        {"from": "00:00", "to": "45:00", "from_period": 1, "to_period": 1},
        {"from": "45:00", "to": None, "from_period": 2, "to_period": 2},
    ]}
    assert abs(player_minutes(full) - 90.0) < 1e-9
    sub = {"positions": [{"from": "63:30", "to": None, "from_period": 2, "to_period": 2}]}
    assert abs(player_minutes(sub) - 26.5) < 1e-9


def test_xt_value_iteration_matches_exact_linear_solve_on_toy_pitch():
    import xt as xtmod
    old = (xtmod.NX, xtmod.NY)
    xtmod.NX, xtmod.NY = 2, 1
    try:
        shots = np.array([[0.0], [10.0]])
        goals = np.array([[0.0], [2.0]])
        starts = np.array([[10.0], [10.0]])
        attempts = np.array([[10.0], [0.0]])
        moves = [((0, 0), (1, 0), "p")] * 10
        iterative = xtmod.fit_xt(moves, shots, goals, starts, attempts)
        exact = xtmod.solve_xt_exact(moves, shots, goals, starts, attempts)
        assert np.max(np.abs(iterative - exact)) < 1e-10
        assert abs(iterative[1, 0] - 0.2) < 1e-9
        assert abs(iterative[0, 0] - 0.2) < 1e-9
    finally:
        xtmod.NX, xtmod.NY = old


def test_failed_moves_leak_value_out_of_xt_chain():
    import xt as xtmod
    old = (xtmod.NX, xtmod.NY)
    xtmod.NX, xtmod.NY = 2, 1
    try:
        shots = np.array([[0.0], [10.0]])
        goals = np.array([[0.0], [2.0]])
        starts = np.array([[20.0], [10.0]])
        moves = [((0, 0), (1, 0), "p")] * 10
        attempts = np.array([[20.0], [0.0]])  # 50% completion
        value = xtmod.fit_xt(moves, shots, goals, starts, attempts)
        assert abs(value[0, 0] - 0.1) < 1e-9
    finally:
        xtmod.NX, xtmod.NY = old


def test_random_placement_does_not_create_material_oof_gain():
    from placement import PLACED, PRE, placement_features
    df = synthetic_shots(3500, seed=7)
    d = placement_features(df.dropna(subset=["end_y", "end_z", "gk_dist"]))
    d = d[d.on_target == 1].copy()
    p0 = cross_fitted_probabilities(d, PRE, seed=7)
    p1 = cross_fitted_probabilities(d, PLACED, seed=7)
    assert log_loss(d.goal, p1) > log_loss(d.goal, p0) - 0.005


def test_finishing_offset_shrinks_small_samples_more():
    from scouting import _player_map
    p_small = np.full(3, 0.2)
    y_small = np.ones(3)
    p_big = np.full(30, 0.2)
    y_big = np.ones(30)
    alpha_small, var_small = _player_map(y_small, p_small, prior_sd=0.75)
    alpha_big, var_big = _player_map(y_big, p_big, prior_sd=0.75)
    assert 0 < alpha_small < alpha_big
    assert var_big < var_small


def test_learning_priority_is_finite_and_normalised():
    from scouting import player_table
    df = synthetic_shots(1800, seed=8)
    df = df.copy()
    df["xg"] = cross_fitted_probabilities(df, FEATURES, seed=8)
    table = player_table(df, min_shots=20)
    assert table.learning_priority.between(0, 1).all()
    assert np.isclose(table.learning_priority.max(), 1.0)


def test_match_report_scores_target_without_using_target_outcomes():
    from match_report import match_frame
    df = synthetic_shots(1800, seed=9)
    mid = int(df.match_id.iloc[0])
    first = match_frame(mid, df=df)
    altered = df.copy()
    altered.loc[altered.match_id == mid, "goal"] = 1 - altered.loc[altered.match_id == mid, "goal"]
    second = match_frame(mid, df=altered)
    assert np.allclose(first.xg.to_numpy(), second.xg.to_numpy())


def test_download_validators_accept_minimal_statsbomb_shapes():
    from download_data import _validate_events, _validate_lineups, _validate_matches
    _validate_matches([{
        "match_id": 1, "match_date": "2000-01-01",
        "home_team": {"home_team_name": "A"}, "away_team": {"away_team_name": "B"},
    }])
    _validate_events([{"type": {"name": "Shot"}}], 1)
    _validate_lineups([
        {"team_name": "A", "lineup": []},
        {"team_name": "B", "lineup": []},
    ], 1)
