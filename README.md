# Football Analytics: xG, xA, xT and Model Evaluation

[![CI](https://github.com/AhmadEblagh205/football-xg-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/football-xg-analytics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I built this to model expected goals, expected assists, post-shot placement, expected threat and player finishing. The real-data mode uses StatsBomb open data for Arsenal's 38 Premier League matches in 2003/04 (competition 2, season 44), which is not the full 380-match season. In v0.8 I changed the evaluation so that headline probabilities come from models that never saw the match being scored, and I added simulated shot worlds where the true effect is known.

## What it does

- fits a geometry xG model (distance, angle, header, set piece) and a richer model with freeze-frame features, scored by five-fold cross-fitting grouped by match;
- credits expected assists from the cross-fitted xG of the assisted shot;
- compares post-shot placement features with pre-strike geometry;
- estimates player finishing offsets with an uncertainty and a learning-priority score;
- solves an xT surface on a 16 x 12 grid and writes leave-one-match-out match reports.

## Results

The checked-in v0.8 results come from simulated shots, because the raw StatsBomb files are not kept in the repository. I have not rerun the real-data models under the grouped protocol, so I do not quote a real-data AUC or xT result for v0.8. The earlier real-data outputs are kept in `reports/legacy_real_data/` as descriptive results fitted in sample.

The robustness experiment uses 900 simulated shots per world, close to the 918-shot Arsenal sample, over 20 worlds. It checks whether the two freeze-frame features help out of sample.

| 900 shots | Median change in AUC | Median change in log loss | Freeze model wins on log loss |
|---|---:|---:|---:|
| No planted freeze-frame signal | -0.0018 | +0.00122 | 10% |
| Planted freeze-frame signal | +0.0340 | -0.02634 | 100% |

A negative change in log loss is better. The extra features slightly hurt when there is no signal and help consistently when there is one. The effect sizes are simulated and say nothing about Premier League football. The test suite has 18 tests and needs no downloaded data.

## How to run

```bash
pip install -r requirements.txt
pytest -q
python src/model.py --synthetic
python src/scouting.py --synthetic
python src/placement.py --synthetic
python src/robustness.py --worlds 20
```

With real data:

```bash
python src/download_data.py              # all available Arsenal league matches
python src/download_data.py --matches 12 # chronological subset
python src/match_report.py --list
python src/match_report.py --match <match_id>
```

After downloading, rerun `model.py`, `scouting.py`, `placement.py` and `xt.py` to regenerate the real-data outputs.

## Repository layout

- `src/shots.py`: event parsing, geometry, freeze-frame features, simulated shot generator
- `src/model.py`: grouped out-of-fold xG, xA and post-shot evaluation
- `src/robustness.py`: repeated simulated feature-value experiment
- `src/scouting.py`: finishing offsets and learning priority
- `src/placement.py`: post-shot placement comparison
- `src/xt.py`: expected threat model with a linear-system check
- `src/minutes.py` and `src/match_report.py`: minutes from lineups, and fixture reports
- `reports/legacy_real_data/`: v0.7 real-data outputs kept for provenance

## Notes

StatsBomb data are downloaded on demand and not redistributed here. Check the terms and attribution at https://github.com/statsbomb/open-data before use. StatsBomb xG is another model, so agreement with it is not validation. The older 0.842 AUC, 0.83 xG correlation, Henry xT/90 and placement figures are legacy in-sample results. The audit changes, the xG, scouting and xT definitions and the placement notes are in docs/METHODOLOGY.md.

## Licence

MIT for the code. StatsBomb event and lineup data remain subject to StatsBomb's own terms and attribution requirements.
