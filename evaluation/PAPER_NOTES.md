# Paper figures and numbers (state 2026-10-16)

Simulation on the original raceline reference (cap 20 m/s), data `02_tracking_v2`, 3 seeds x 10 evaluated laps; corrected
simulator (branch commit c21a1e5). Hardware: CPM Lab 2026-10-15, frozen config `cpm_lab_controller_raceline_pp_eval1014`,
3 runs per tyre set and variant. "matched" / "mismatched" = oracle / inverse runs.

| Paper | File |
|---|---|
| Fig. 1 (unchanged) | `01_prediction/prediction_error.pdf` |
| Fig. 2 | `02_tracking/tracking_error_l_shape.pdf` (top: DPC pooled vs NMPC; bottom: mismatched / pooled vs matched) |
| Fig. 3 | `02_tracking/implied_friction_l_shape.pdf` (evidence-weighted implied friction) |
| Fig. 4 (hardware) | `04_hardware/hardware.pdf` (lateral error vs matched along the lap) |
| Table I | `02_tracking/table_tracking.tex` (incl. the Nürburgring column; its figure `tracking_error_Nuerburgring.pdf` is dropped) |

Numbers for the text (cm unless stated):
- Tracking, lap: pooled 13.7, matched 14.4, mismatched 15.0, NMPC 15.6. mu 0.5: pooled 12.1 vs NMPC 19.3; mu 1.1: NMPC best
  (12.1 vs pooled 14.0, mismatched 19.2). Seed-to-seed spread <= 0.2 (lap).
- Prediction (rms over the 10-step horizon, median): pooled 3.5 / 8.5 / 4.8, matched 3.0 / 8.2 / 4.0, mismatched 6.1 / 8.6 /
  8.6 at mu 0.5 / 0.8 / 1.1.
- Pooled tracks like the better single dataset: where matched and mismatched differ by > 2 cm (54% of the lap; mismatched is the
  better one on half of those), pooled is closer to the better one on 76% (2.1 above it, the worse one 8.0 above).
- Along the lap vs matched (Fig. 2 bottom): mismatched +0.6, pooled -0.8 (lap mean); mu 1.1 zone up to +20 for mismatched.
- Implied friction (Fig. 3): 0.77 / 0.88 / 0.90 in the mu 0.5 / 0.8 / 1.1 zones.
- Nürburgring (lap): pooled 8.8, NMPC 8.4; largest DPC deviation at the hairpin (curvature 0.091 1/m, twice the L-shape data).
- Hardware, slippery tyres: lateral pooled 2.0, matched 1.9, mismatched 4.0; prediction (rms, 20 steps) 1.28 / 1.27 / 2.62;
  lap 9.62 / 9.51 / 9.30 s. Normal tyres: lateral 1.3 / 1.5 / 1.0, prediction 0.90 / 0.85 / 1.19, lap 10.17 / 9.84 / 10.35 s
  (the mismatched data tracks tighter but is slowest). Along the lap vs matched: slippery mismatched +1.44, pooled +0.12.
- Why the sim separates the datasets less than the lab: in the simulator's tyre model friction scales the peak force only
  (cornering stiffness K_y = F_z p_ky1, independent of mu), so friction changes the motion only near saturation; the raceline
  profile uses at most about 0.5 / 0.4 of the grip in the mu 0.8 / 1.1 zones. Real tyre sets also differ in stiffness.

Wording suggestions from the earlier robustness checks: `exploratory/PAPER_CHANGES.md` (older, check against these numbers).
