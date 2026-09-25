# Exploratory results, 2026-09-25/26

Control experiments on the implicit-friction-selection claim, run while preparing the
ECC submission. Exploratory: not (yet) part of the paper's evaluation pipeline.
All runs use the `l_shape` track, 2-3 laps, first lap discarded.

## 1. Uniform friction: is the selection driven by friction or by the track?

`uniform_friction/` — the same geometry and reference speed profile, `use_dyn_friction`
off, one run per constant friction. Any structure left in the implied friction cannot
come from the spatial arrangement of the friction zones.

Separation = difference in implied friction between the mu=0.5 and mu=1.1 runs,
as a fraction of the true difference of 0.6.

| regime | separation | % of true |
|---|---|---|
| all samples | +0.165 | 27% |
| straights (a_lat < 2.8) | +0.039 | 6% |
| corners (a_lat > 3.5) | +0.298 | **50%** |

The mechanism is real but partial and excitation-dependent. Note the asymmetry:
mu=0.5 is recognized (+0.19 from the neutral 0.8), mu=1.1 is not (-0.02).

Also: at constant friction the implied friction still swings by roughly +-0.25 along
the lap, so part of the pattern in the paper's Fig. 5 is geometry, not friction.

**Bin by lateral acceleration, not speed.** The straights are the fastest part of the
track and carry no friction information; binning by speed inverts the picture.

## 2. Why the asymmetry: observability, not data coverage

Dataset check (both subsets, `data/dpc_experiments/02_tracking/`):

- speed and lateral-acceleration distributions are nearly identical (max a_lat 4.4 vs 4.3)
- above 3 m/s^2 the subsets are balanced 50/50 -> the corner result is not a coverage artifact
- neither subset contains anything above ~4.4 m/s^2

At those maneuvers mu=0.5 runs at ~90% of its friction limit (4.9 m/s^2) and is therefore
identifiable, while mu=1.1 stays at ~41% of its limit (10.8 m/s^2) and behaves nearly
kinematically. The selection identifies the *binding* regime; high friction is only
inferable by exclusion.

Consequence for the hardware experiments: the maneuvers must reach a substantial
fraction of the *slippery* tires' limit, otherwise no selection can be observed.

## 3. Regularization sweep (lambda_g)

| lambda_g | tracking err (0.5 / 1.1) | corner separation |
|---|---|---|
| 0.1 | 16.5 / 5.3 (diverged) | -18% |
| 1.1 (default) | 0.16 / 0.21 | **51%** |
| 10.0 | 23.1 / 0.21 (diverged) | 39% |

Only the default tracks in both conditions. Selection appears at the regularization
where the controller works; there is no "more regularization, more selection" trend.

## 4. Method variants that do not help

All at 2 laps, corner separation vs the 51% baseline:

| variant | corners | tracking |
|---|---|---|
| baseline | **51%** | 0.16 / 0.21 |
| group-sparse penalty on subsets (lambda_group=1.0) | 48% | 0.16 / 0.21 |
| regime-persistence prior (lambda_persist=0.1) | 22% | 0.16 / 0.21 |
| explicit hard selection by window residual | 0% | 14.9 / 0.21 |

- Group sparsity concentrates the weights but adds no discrimination: the limit is
  observability, not the regularizer. (An exact group-lasso is also an SOCP; ECOS and
  CLARABEL fail on it and SCS is far too slow. The version used here is a reweighted-l2
  surrogate, which stays a QP.)
- The persistence prior suppresses the signal: pulling toward the previous subset weight
  also prevents evidence from accumulating.
- Hard selection by initialization-window residual is at chance (55% / 46% correct) and
  halves the candidate set, which diverges the mu=0.5 run. The regime information lives
  in how the coefficients distribute, not in a per-subset matching residual.

Implemented in `external/dpc` as `controller.lambda_group`, `controller.lambda_persist`
and `controller.explicit_selection`, all inert at their defaults.

## 5. Offline distinguishability across output channels

`observability/cross_prediction_channels.py` — no controller involved. Measures
eps = (error using the wrong friction subset) - (error using the matching subset),
i.e. how informative the initialization window is about friction.

Mismatched/matched error ratio, corner windows only:

| output channels | mu=0.5 | mu=1.1 |
|---|---|---|
| base (x, y, v, yaw) | 1.07 | **1.89** |
| + yaw rate | 1.03 | 1.43 |
| + a_lat = v * yaw_rate | 0.83 | 1.82 |
| + both | 0.88 | 1.69 |

- eps is positive in corners and ~zero over all windows: independent confirmation of the
  excitation dependence, without the controller or the eta diagnostic.
- No channel augmentation helps; the base set is best. Extra sensors are not the fix.
- **Open question:** offline, mu=1.1 is the more distinguishable regime (1.89 vs 1.07),
  which is the opposite of the closed-loop asymmetry in section 1. Unresolved.

## Caveats

- 2-3 laps per run, one track, single seed.
- The eta diagnostic weights subsets by ||Y_f,j g_j||, so subsets with larger-magnitude
  futures (higher speed) get more weight mechanically. A ||g_j||-based version has not
  been checked; it needs g exposed from the adapter.
- Cross-variant comparison in section 5 also changes the error metric and the retrieval
  metric, so the ratios are indicative rather than exact.

## Reproduce

```bash
.venv/bin/python evaluation/exploratory/uniform_friction/run_uniform_friction.py --laps 3
.venv/bin/python evaluation/exploratory/uniform_friction/analyze_uniform_friction.py
.venv/bin/python evaluation/exploratory/observability/cross_prediction_channels.py
```

The runner drives `DPC.py` through environment overrides (`DPC_MAP_NAME`,
`DPC_USE_DYN_FRICTION`, `DPC_CONSTANT_FRICTION`, `DPC_NUMBER_OF_LAPS`, `DPC_LOG_SUFFIX`,
`DPC_VELOCITY_MULTIPLIER`, `DPC_HEADLESS`, `DPC_HYDRA_OVERRIDES`), each defaulting to the
value in the file, so a plain `python DPC.py` is unchanged.
