# Simulation DPC data, friction-aware profile (2026-10-16)

Pure pursuit on l_shape at constant friction, one continuous run per friction, velocity profile multiplier per lap 0.5, 0.6,
0.7, 0.8, 0.9, 1.0, 1.05, applied acceleration as input (as v2).
- `pure_pursuit_0_5`: the v2 mu 0.5 run (02_tracking_v2, on the raceline profile; at mu 0.5 the friction-aware profile is
  at most 1.1 m/s slower, in braking zones, so this run covers it).
- `pure_pursuit_1_1`: recorded on the mu 1.1 profile `maps/l_shape/waypoints/l_shape_mintime_l720_mu1.1.csv`
  (`evaluation/02_tracking/friction_aware_profile.py <out> 1.1`) with `DPC_collect_data.py` (DPC_MAP_NAME=l_shape
  DPC_USE_DYN_FRICTION=0 DPC_CONSTANT_FRICTION=1.1 DPC_NUMBER_OF_LAPS=7 DPC_SPEED_LAPS=0.5,0.6,0.7,0.8,0.9,1.0,1.05
  DPC_ABORT_ERROR=15 DPC_WPT_PATH=<that file>); all laps completed, up to 23.8 m/s.
The evaluation (evaluation/02_tracking/run_eval.sh) drives l_shape on `l_shape_mintime_l720_mu_aware.csv` (the friction map).
