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

## Notes moved from the README

Audit changes in v0.8. The earlier release reported AUC, log loss, calibration and the geometry against freeze-frame comparison on the same shots used to fit the model. Those numbers are kept under reports/legacy_real_data/ as descriptive outputs and are not v0.8 validation metrics. The current pipeline uses five-fold stratified group cross-fitting by match ID, so every headline probability comes from a model that did not train on that match. The same out-of-fold probabilities feed calibration, xA and player evaluation. Match reports fit on every other match before scoring the selected fixture.

Two implementation fixes. The downloader previously fetched event files but not the lineup files needed for per-90 minutes. The old set_piece flag treated almost every non-regular play pattern as a set piece, including goal kicks and keeper restarts, and is now restricted to explicit corner, free-kick and throw-in patterns. The old test suite needed raw StatsBomb files that a clean clone does not have, so three tests failed; the current suite uses synthetic or temporary fixtures and has 18 passing tests with no download.

Model details. The geometry model uses distance to the goal centre, the angle between the posts, a header indicator and an explicit set-piece indicator, for non-penalty shots. The richer model adds the number of opposing defenders inside the shot cone and the goalkeeper distance from the goal centre. src/model.py fits full-data coefficients only for interpretation and plots, and reports ROC AUC, log loss, Brier score and equal-frequency expected calibration error from grouped out-of-fold probabilities. Correlation with StatsBomb xG, when available, is an agreement diagnostic and not ground truth.

Scouting. The old table used an ad-hoc goals/xG shrinkage ratio with a variance expression that was not a coherent posterior variance. v0.8 uses a regularised finishing offset, logit P(goal) = logit(xG) + alpha_j with alpha_j ~ N(0, sigma_alpha^2), a MAP estimate plus a Laplace approximation. The learning_priority score measures how much a typical future match's shots would reduce the variance of that offset. It is an information diagnostic, not a transfer recommendation or valuation.

xT. The pitch is a 16 x 12 grid and V = r + QV. Failed passes stay in the attempted-move denominator, so lost possession leaks probability out of the system. The model is solved by value iteration and by the linear system (I-Q)V = r, and the tests require the two to agree on a toy pitch.

Placement. The checked-in synthetic placement run is a negative control with random goalmouth coordinates. The earlier real-data finding that placement looked flat while goalkeeper distance looked more useful was in sample and is not audited evidence until the workflow is rerun on downloaded data.

Not established by this repository: that a freeze-frame model reaches 0.842 AUC out of sample, that StatsBomb xG is ground truth, that learning priority is a recruitment model, that the old placement result generalises, that xT is causal, or that the 38 matches are a whole Premier League season. The older 0.842 AUC, 0.83 xG correlation, Henry xT/90 and placement outputs remain legacy descriptive results until the real data are rerun.
