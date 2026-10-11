# Segment sampling diagnostics (2026-10)

Everything in this folder is diagnostic and can be deleted as a unit. It backs the findings handed to the
lab in the dpc repo, `evaluation/multisurface_dpc/SIM_FINDINGS_FOR_LAB.md` (branch
`multisurface-racing-with-dpc-lab`), which also describes segment sampling itself.

Outside this folder, the diagnostics only need:
- `DPC.py`: env overrides `DPC_VELOCITY_MAX`, `DPC_INIT_HORIZON`, `DPC_MEAS_NOISE`, `DPC_ORACLE`,
  `DPC_ABORT_ERROR`, `DPC_MAX_TIME`, `DPC_NO_LIDAR`, and the logged plan `pred_xy` (promise gap)
- `regulators/dpc_adapter.py`: `DPC_DATA_DIR`, and `dataset.reuse_cached_merge=true` (keeps the original
  dataset with the current dpc branch; not a diagnostic)
- `data/dpc_experiments/02_tracking_unpaired/` (recorded by `closed_loop/make_unpaired_dataset.sh`)

## Files

| file | what it answers |
|---|---|
| `closed_loop/run.sh` | runs sampling variants on l_shape (multi-friction + uniform 0.5/1.1), logs to `logs/` (ignored) |
| `closed_loop/compare.py` | the comparison table below from those logs |
| `closed_loop/make_unpaired_dataset.sh` | records the unpaired mu=0.5 dataset |
| `offline/offline_tuning.py` | replays segment selection + the controller's predictor on recorded windows: matching share and prediction error per (G, k) |
| `offline/offline_qp_replay.py` | runs the real controller (sampling + QP) on recorded windows |
| `design/predictive_evidence.py` | does predicting the recent past (memory M) beat fitting the current window? |
| `design/segment_evidence.py` | does the window fit work without labels, with short segments as groups? |
| `design/selection_tests.py` | top-k purity of segment selection (1), labelled upper bound vs. fit freedom (2), purity vs. tracking proxy (3), lab-like datasets (`lab`) |

Batch runs need `OPENBLAS_NUM_THREADS=1` (set in `run.sh`); add `DPC_NO_LIDAR=1 DPC_ABORT_ERROR=5` to the
env of a variant (12% faster, failing runs end at once).

## Results (l_shape, single runs)

Closed loop (`compare.py`). The segment ridge default is now 0.01; rows marked ridge 0.1 need
`++controller.path_sampling.segment_ridge=0.1` to reproduce. Uniform: separation of implied friction in corners, as share of the true 0.6;
multi: correlation of implied friction with true friction in corners. "QP" = from the QP usage.

| variant | tracking multi [m] | uniform corners QP | multi corners QP |
|---|---|---|---|
| contextual sampling (baseline) | 0.191 | 45% | 0.81 |
| segment 2.5 s x 4, ridge 0.1 (`_slide25x4`) | 0.187 | 78% | 0.87 |
| segment 4 s x 6, ridge 0.1 (`_slide40x6`) | 0.192 | 73% | 0.87 |
| baseline, input slack (`_islack`) | 0.193 | 44% | 0.82 |
| baseline, T_ini = 2 (`_tini2`) | 0.453 | 0% | 0.29 |
| segment 4 s x 6, ridge 0.1, T_ini = 2 | 0.287 | 55% | 0.83 |
| baseline, unpaired data (`_unpaired`) | 0.166 | 14% | 0.22 |
| segment 4 s x 6, ridge 0.1, unpaired data | 0.178 | 19% | 0.05 |
| segment 2.5 s x 4, ridge 0.01 (`_slide25x4_r001`) | 0.212 | 94% | 0.93 |
| segment-gated contextual, whole recording (`_gate_run`) | 0.192 | 98% | 0.95 |
| segment-gated contextual, +-1 lap (`_gate_lap`) | 0.185 | 99% | 0.93 |
| oracle: only the matching dataset (`_oracle`) | 0.192 | 100% | 0.98 |

- The segment fit needs freedom: ridge 0.1 blurs it (pool ~60% matching), ridge 0.01 identifies the
  matching pass (~90%). Pure segment pools track 2-3 cm worse (few passes, little diversity); gating
  contextual sampling by the best segment keeps the selection and the tracking.
- Correct data (oracle, gated) lowers the promise gap from 0.17 to 0.14 m but does not improve tracking
  overall (uniform mu=0.5 0.18 vs 0.16 m, mu=1.1 0.18 vs 0.21 m, multi equal).
- Speed profile 1.1x / 1.2x / 1.4x at mu=0.5 (`_v11`, `_v12`, `_v14`, `_gate_*`, `_oracle_v11`): the car
  leaves the track at the first corner with any selection, oracle included; the baseline shifts to mu=1.1
  data just before (false promise). Infeasible reference, not selection.
- 5 s x 2 separates most (95%) but fails at uniform mu=0.5; measurement noise of 5 cm / 0.01 rad breaks
  the controller at mu=0.5 with any sampler.
- Ranking segments by the QP's own consistency cost instead of the window fit: better tracking, less
  separation (dropped).
- Evidence-weighted contextual sampling (dpc `ad280ea`) with forgetting 0.84: posterior inverted for ~0.5 s
  after friction changes, multi corr 0.65 (dropped).

Offline:
- Window fit of the current window (no memory) beats predicting the last 5-20 steps: corners 95% / 79%
  correct vs 68% / 65% at M = 20; straights at chance either way (`design/predictive_evidence.py`).
- Plain closeness over a longer history (5-30 steps) is at chance: trajectory shape dominates, friction
  is a small residual (script removed; result only).
- Label-free segments of 2.5 s: best segment from the matching friction 83% / 89% in corners, close to
  labels 96% / 80% (`design/segment_evidence.py`).
- The matching share of the selected pool (`offline_tuning.py`) predicts the closed-loop ranking of the
  friction separation; neither offline script predicts the failures at low friction.
- Labelled group fit (`selection_tests.py 2`): 96-98% correct in corners with 40-80 columns and ridge
  0.01, ~70% with 20 columns and ridge 0.1 (the lab evidence-weighting defaults). Offline, the run decision
  beats the segment ranking; in closed loop it is the other way round.
