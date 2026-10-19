# Simulation DPC data, v2 (2026-10-15)

Pure pursuit on l_shape at constant friction (mu 0.5 / 1.1), one continuous run per friction, velocity profile multiplier
per lap 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05 (the v1 data in 02_tracking: 0.5 ... 1.0). Recorded with
`DPC_collect_data.py` (DPC_MAP_NAME=l_shape DPC_USE_DYN_FRICTION=0 DPC_CONSTANT_FRICTION=<mu> DPC_NUMBER_OF_LAPS=7
DPC_SPEED_LAPS=0.5,0.6,0.7,0.8,0.9,1.0,1.05 DPC_ABORT_ERROR=15) and the applied acceleration as input
(helpers/applied_accel.py; reproduces the v1 data exactly).
At mu 0.5 pure pursuit loses the car in the x1.05 lap (second slide after ~300 m); that recording ends at the last row
before the abort with a tracking error below 1 m (it keeps the first, recovered slide). mu 1.1 completes all laps.
