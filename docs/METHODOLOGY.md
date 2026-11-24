# Methodology and audit decisions

## 1. Evaluation unit

Shots from the same football match are not treated as independent train/test units. Model validation uses stratified group cross-fitting with `match_id` as the group, so a held-out shot's match does not appear in its training fold.

The full-data logistic model is still fitted for coefficient inspection and smooth xG curves, but those in-sample predictions are not used for headline AUC, log loss, Brier score or calibration.

## 2. Feature-complexity experiment

The known-truth generator produces two worlds:

- `freeze_strength=0`: blockers and goalkeeper distance are irrelevant conditional on geometry;
- `freeze_strength=1`: both variables enter the true goal logit.

Geometry and freeze-frame models are then compared using the same grouped out-of-fold splits. This is a falsification test: the richer model should not win systematically when its extra terms are absent from the DGP.

## 3. StatsBomb xG

StatsBomb xG is an external model prediction, not an observed truth. Correlation with it is reported only as agreement. Goal outcomes remain the observed binary response for discrimination and proper scoring rules.

## 4. xA

An assisted shot contributes its grouped out-of-fold xG to the assisting player. This avoids summing probabilities fitted on the same match they describe.

## 5. Placement

Placement models are evaluated on on-target shots only and use grouped out-of-fold scoring. The legacy 2003/04 placement result was fitted and scored on the same 316 shots, so it remains historical rather than audited evidence.

## 6. Finishing offsets

For player j and shot i:

`logit(q_ij) = logit(xG_ij) + alpha_j`

with `alpha_j ~ N(0, prior_sd^2)`. The MAP estimate is found numerically. A Laplace variance uses posterior precision

`1/prior_sd^2 + sum(q_ij * (1-q_ij))`.

The information-priority score approximates how much that variance would fall after one typical match's expected shot volume. It is designed to illustrate information allocation, not to rank transfer targets.

## 7. xT

For each zone z:

`V(z) = shoot_p(z) * goal_p(z) + sum_z' Q(z,z') V(z')`.

`Q` contains unconditional successful-move probabilities. Failed passes stay in the attempted-move denominator, so each row may sum to less than the probability of choosing to move. The model is solved both by iteration and by `(I-Q)^-1 r` for independent numerical validation.

## 8. Data provenance

StatsBomb raw event and lineup JSON files are downloaded directly from the open-data repository and kept under `data/raw/`, which is gitignored. The checked-in CI evidence uses synthetic fixtures so a clean clone does not depend on a network call or redistribution of raw data.
