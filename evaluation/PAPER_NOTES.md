# Paper figures and numbers (state 2026-10-17)

Simulation on the original raceline reference (cap 20 m/s), data `02_tracking_v2`, 3 seeds x 10 evaluated laps; corrected
simulator (branch commit c21a1e5). Hardware: CPM Lab 2026-10-15, frozen config `cpm_lab_controller_raceline_pp_eval1014`,
3 runs per tyre set and variant. "matched" / "mismatched" = oracle / inverse runs.

| Paper | File |
|---|---|
| Fig. 1 | `01_prediction/prediction_error.pdf` (open-loop prediction on held-out windows of the random-excitation data `data/dpc_experiments/01_prediction_v2`, mu 0.5 and 1.1, recorded 2026-10-18 (see its README), 300 contextual columns of pooled / matched / mismatched; `prediction_results_random.json`, usage in `visualize_prediction_errors.py`) |
| Fig. 2 | `02_tracking/tracking_error_l_shape.pdf` (top: DPC pooled vs NMPC in cm; middle / bottom: tracking and prediction error of mismatched / pooled vs matched) |
| Fig. 3 | `02_tracking/implied_friction_l_shape.pdf` (evidence-weighted implied friction) |
| Fig. 4 (hardware) | `04_hardware/hardware.pdf` (lateral error vs matched along the lap) |
| Table I | `02_tracking/table_tracking.tex` (incl. the Nürburgring column; its figure `tracking_error_Nuerburgring.pdf` is dropped) |
| Table II (hardware) | numbers below (hardware, per tyre set) |

Numbers for the text (cm unless stated):
- Prediction, random-excitation data (Fig. 1, T_ini 10 = 1 s as on the hardware, median rms over the horizon, mu 0.5 / 1.1,
  1996 windows each): pooled 18.8 / 24.6, matched 12.7 / 23.6, mismatched 42.5 / 51.7; paired median vs matched: pooled +4.6 / +1.4,
  mismatched +26.2 / +22.5; pooled closer to matched than to mismatched on 73 / 70% of the windows.
  With T_ini 5 (the closed-loop simulation's window): pooled 22.4 / 27.0, matched 12.6 / 27.6, mismatched 48.0 / 56.5.
  Why pooled lags matched at mu 0.5 (200-window diagnostic): the active set is balanced (50-60% own regime), but the weight on the
  own regime (1'g) is only 0.65 (T_ini 5) / 0.73 (T_ini 10); lambda_sigma x100 or 1000 columns help little, T_ini 20 not more.
  Grip use (share of samples above 70% of mu g): random 68% (mu 0.5) / 57% (mu 1.1), raceline 6% (mu 0.5) / 0% (mu 1.1).
  The earlier random data in 01_prediction (labelled mu 0.3 / 1.0; their stored configs all say mu 1.1, so the friction is
  undocumented) separate by speed (median 29.6 vs 13.8 m/s): weight on the own regime 0.99 / 0.93; pooled 17.6 / 18.4, matched
  13.2 / 17.2, mismatched 58.7 / 63.0 (T_ini 5).
- Open item: closed-loop simulation with T_ini 10 (currently 5), to be run.
- Prediction, closed loop (median per step): pooled 3.5 / 4.8, matched 3.0 / 4.0, mismatched 6.1 / 8.6 at mu 0.5 / 1.1.
- Prediction, open loop on the raceline data (not in the paper; median rms position error over the horizon, mu 0.5 / 1.1): matched 4.7 / 4.1, mismatched
  5.9 / 5.1, pooled 5.0 / 4.1, pooled random 10.4 / 9.5, pooled full 9.1 / 9.0 (full matched 9.4 / 8.8). 829 / 881 windows.
- Prediction error along the lap vs matched (Fig. 2 bottom): pooled +0.1 (lap mean), +1.1 to +1.3 in the first 30 m after the
  transitions at 0 / 271 / 475 m; mismatched +1.8, up to +8.6 at 595 m. Largest pooled tracking deficit 5.0 at 595 m.
- Lap times in the sim: all four controllers 47.8 to 47.9 s (same speed profile).
- Tracking, lap: pooled 13.7, matched 14.4, mismatched 15.0, NMPC 15.6. mu 0.5: pooled 12.1 vs NMPC 19.3; mu 1.1: NMPC best
  (12.1 vs pooled 14.0, mismatched 19.2). Seed-to-seed spread <= 0.2 (lap).
- Prediction (rms over the 10-step horizon, median): pooled 3.5 / 8.5 / 4.8, matched 3.0 / 8.2 / 4.0, mismatched 6.1 / 8.6 /
  8.6 at mu 0.5 / 0.8 / 1.1.
- Pooled tracks like the better single dataset: where matched and mismatched differ by > 2 cm (54% of the lap; mismatched is the
  better one on half of those), pooled is closer to the better one on 76% (2.1 above it, the worse one 8.0 above).
- Along the lap vs matched (Fig. 2 bottom): mismatched +0.6, pooled -0.8 (lap mean); mu 1.1 zone up to +20 for mismatched.
- Implied friction (Fig. 3): 0.77 / 0.88 / 0.90 in the mu 0.5 / 0.8 / 1.1 zones; readable (two predictions > 1 cm apart) on 99% of
  the lap; on the 20% of the lap where they differ most (> 6.5 cm) 0.77 / 0.89 / 0.96.
- Infeasible reference (`run_infeasible_reference.sh`, constant mu, 2 laps): speed profile x1.1 at mu 0.5: pooled, matched and
  mismatched all leave the track in the first corner (50 / 54 / 68 m); prediction error before that (median, first 60 m) 52 / 34 /
  50 cm. At x1.0 all complete (2.8 / 2.4 / 5.4 cm); at mu 1.1 the x1.1 profile is completed by all (4.7 / 3.1 / 9.3 cm).
- Nürburgring (lap): pooled 8.8, NMPC 8.4; largest DPC deviation at the hairpin (curvature 0.091 1/m, twice the L-shape data).
- Hardware, slippery tyres: lateral pooled 2.0 (1.5-2.2 over the 3 runs), matched 1.9 (1.5-2.2), mismatched 4.0 (2.9-4.7);
  prediction (rms, 20 steps) 1.28 / 1.27 / 2.62; lap 9.62 / 9.51 / 9.30 s. Normal tyres: lateral 1.3 (1.0-1.6) / 1.5 (1.3-1.6) /
  1.0 (0.9-1.2), prediction 0.90 / 0.85 / 1.19, lap 10.17 / 9.84 / 10.35 s
  (the mismatched data tracks tighter but is slowest). Along the lap vs matched: slippery mismatched +1.44, pooled +0.12.
- Why the sim separates the datasets less than the lab: in the simulator's tyre model friction scales the peak force only
  (cornering stiffness K_y = F_z p_ky1, independent of mu), so friction changes the motion only near saturation; the raceline
  profile uses at most about 0.5 / 0.4 of the grip in the mu 0.8 / 1.1 zones. Real tyre sets also differ in stiffness.

Wording suggestions from the earlier robustness checks: `exploratory/PAPER_CHANGES.md` (older, check against these numbers).
