# Changelog

## 0.8.0, defensibility audit
- Replaced in-sample headline xG metrics with stratified group cross-fitting by match ID.
- Preserved the old real-data reports under `reports/legacy_real_data/` as historical descriptive outputs rather than audited validation evidence.
- Added repeated known-truth worlds showing that freeze-frame features help only when their signal is genuinely present; at 900 shots the median held-out AUC change is -0.0018 with no planted signal versus +0.0340 when the signal is planted.
- Corrected the downloader so it fetches lineup JSON as well as events, making the advertised per-90 workflow reproducible.
- Tightened the set-piece definition so goal-kick/keeper restarts are not silently labelled set pieces.
- Rebuilt player finishing as a regularised log-odds offset with Laplace uncertainty and an explicit expected-variance-reduction learning priority.
- Made match reports leave-one-match-out: the selected fixture no longer helps fit its own xG probabilities.
- Added an independent linear-system solution for xT and convergence-based value iteration.
- Reworked placement evaluation to use grouped out-of-fold probabilities and added a synthetic negative control.
- Removed clean-clone dependence on absent raw StatsBomb files; CI now runs entirely offline.
- Expanded the offline test suite from 10 tests (3 failing in a clean clone) to 18 passing tests.
- Added `docs/METHODOLOGY.md` and explicit provenance/claim boundaries.

## 0.7.0, update round 9
A match report generator: pick any of the 38 matches and get the xG race by minute, a shot map with goals ringed, and a written summary of goals against xG per side, one command from events to story.

## 0.6.0, update round 8
Added the shot placement study with its surprise: on 316 real on-target shots, placement adds under half an AUC point over pre-strike geometry (0.811 to 0.815) because placement is endogenous, close shots go low and central and score anyway, while keeper distance adds more than placement does. The keeper view surface ships nearly flat, which is the finding.

## 0.5.0, update round 7
Added expected threat: a 16 by 12 grid whose zone values are the fixed point of shoot value plus moved value, with possession losses draining the chain (failed passes normalise the transition matrix), an xT surface figure, and per 90 xT added rankings. Henry 0.41 and Pires 0.39 xT added per 90 lead the season.

## 0.4.0, update round 6
Per 90 rates from the lineup files: a minutes parser over every position span (validated by Lehmann's 3,420 minutes, a keeper plays them all), joined into the scouting table with a 900 minute floor and a per 90 xG chart. Henry runs at 0.46 non penalty xG and 0.62 non penalty goals per 90; on a per 90 basis Bergkamp matches Ljungberg despite far fewer minutes.

## 0.3.0, update round 5
Ran the promised full season test: all 38 matches, 918 shots. The freeze frame model now beats the geometry only model on discrimination (AUC 0.842 against 0.838) and on agreement with StatsBomb's xG (0.83 against 0.74); keeper distance firms up while the blocker count stays marginal, collinear with the geometry. Season scale xA: Henry 9.5, Pires 6.2, Bergkamp 4.9.

## 0.2.0, update round 2
Added freeze-frame features (defenders in the shot cone, keeper distance) as a second generation model. On the 12 match subset discrimination is flat but agreement with StatsBomb's own xG rises from 0.82 to 0.87; the full season download is the stated next test.

## 0.1.0
Initial release.
