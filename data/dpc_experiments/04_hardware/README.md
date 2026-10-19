# CPM Lab data (2026-10-14) and evaluation runs (2026-10-15)

`vehicle_1_lab_pp1014`: 1:18 vehicle (vehicle_1) in the CPM Lab, pure pursuit on the raceline `raceline/figure8_lat1.8.csv`
(12.04 m, speed profile for a_lat 1.8 m/s², 0.97-1.20 m/s), 50 Hz, one 100 s run per speed gain 0.6 ... 1.2 and tyre set:
`S_pp_g*` low-friction (slippery) tyres, `N_pp_g*` normal tyres (recorded after a bearing change, 0% IPS fixes rejected);
14 runs, 70530 samples. Inputs: the normalised steering and throttle commands sent to the vehicle ([-1, 1]); outputs:
position, speed, yaw. Dataset labels in the pooled data: 0-6 = N_pp_g0.6 ... 1.2, 7-13 = S_pp_g0.6 ... 1.2.

`eval_runs/<run>`: the 18 evaluation runs (3 per tyre set and variant, low-friction block first, rotated order), each with
the resolved controller configuration (`config.yaml`; `controller.path_sampling.oracle_labels` selects the matched /
mismatched data, absent = pooled), the data folders used (`dataset.txt`) and `log.npz` (time, state, sent input, reference,
progress, the controller's 20-step prediction (normalised, aligned to the current state), the labels of the sampled columns,
compute time). Evaluation setting: T_ini 10, N 20, 100 columns, lambda_g 300, lambda_sigma 10, slack on the past inputs
(controller.input_slack, penalty lambda_sigma), speed gain 1.1, rolling start (the data-collection pure pursuit drives the
first 1 s; the data hold no standstill). The controller is deepc 16fe5f4 (external/dpc) run by the CPM Lab's ROS 2 node
(not public), which adds the raceline reference, the rolling start, the IPS state input and sends the commands at 50 Hz.

Results: `evaluation/04_hardware/extract_lab_runs.py` -> `hardware_runs.json` -> `plot_hardware.py`.
