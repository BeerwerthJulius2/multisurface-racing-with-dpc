# Exploratory results, 2026-09-25/27

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

## 6. Robustness: is the effect real, or a lucky scenario? (2026-09-27)

`observability/pooled_influence.py` — the offline analogue of the paper's implied-friction
figure: predict a window from the POOLED dataset and attribute the predicted future to the
subsets via eta. Unlike the closed-loop version it admits proper controls.

| check | corner separation | verdict |
|---|---|---|
| real labels, 5 seeds | **+23.2% +- 1.1%** | the effect |
| shuffled labels, 5 seeds | -0.1% +- 1.2% | control clean |
| placebo (one run split in half, same friction) | none | control clean |

~20 sigma from the control, so not a lucky draw.

### Paired coverage is a condition

| candidate pool | corner separation |
|---|---|
| same-track-position trajectories excluded | +4% |
| same-track-position trajectories kept | +22% |

Discrimination needs the dataset to contain *comparable trajectories under both frictions at
the current operating point*. The mechanism chooses between near-identical maneuvers recorded
under different conditions; it does not recognize friction in the abstract. The shuffled
control with twins kept gives +1%, so this is not an artifact of the twins themselves.

**For the hardware campaign: drive the same maneuvers with both tire sets.** If the two
conditions are recorded on different trajectories, the mechanism cannot show.

### Parameter trends (offline, twins kept)

| parameter | values | corner separation |
|---|---|---|
| T_ini | 3 / 5 / 10 | 11% / 22% / **45%** |
| lambda_sigma | 1.01 / 10.1 / 101 | 9% / 22% / **61%** |
| candidates | 50 / 100 / 300 | 40% / 22% / 17% |

- T_ini and lambda_sigma behave as Section III predicts: a longer window accumulates more
  evidence, and a larger slack penalty makes wrong-regime columns expensive. The
  lambda_sigma claim in the Conditions paragraph was previously untested.
- More candidates *reduce* discrimination: with more columns the wrong subset reconstructs
  the window more easily. Contextual sampling therefore aids selection, not only tractability.
- The sharpening does not transfer to closed loop: at lambda_sigma=101 the mu=0.5 run
  diverges (tracking error 23.2). The operating point is a compromise.

### Other geometries

| track | corner separation | note |
|---|---|---|
| l_shape (baseline) | 51% (closed loop) | a_lat > 3.5 |
| DualLaneChange, velocity 0.8 | 12%, correct ordering | a_lat > 2.4, weak replication |
| Nuerburgring, velocity 0.6 | 1% | a_lat > 1.2 only: never leaves the linear region |

Nuerburgring cannot test the mechanism: reaching ~3.5 m/s^2 on those radii needs ~30 m/s,
while the dataset ends at 20.1 m/s. Either outside the data or below the excitation threshold.
At full speed on DualLaneChange the mu=0.5 run spins out, hence velocity 0.8.

## Caveats

- 2-3 laps per run, one track, single seed.
- The eta diagnostic weights subsets by ||Y_f,j g_j||, so subsets with larger-magnitude
  futures (higher speed) get more weight mechanically. A ||g_j||-based version has not
  been checked; it needs g exposed from the adapter.
- The placebo sits at 0.745 rather than the neutral 0.80, i.e. splitting one run in half
  produces a bias of about -0.05. An order of magnitude below the effect, but unexplained.
- Not checked: random vs contextual sampling in closed loop.
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
