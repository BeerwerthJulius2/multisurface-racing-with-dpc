# CPM Lab data (2026-10-14)

1:18 vehicle (vehicle_1) in the CPM Lab, pure pursuit on the raceline `figure8_lat1.8` (a_lat 1.8 m/s²), 50 Hz, one 100 s run
per speed gain 0.6 ... 1.2 and tyre set: `S_pp_g*` slippery tyres, `N_pp_g*` normal tyres (recorded after a bearing change,
0% IPS fixes rejected). Dataset labels in the pooled data: 0-6 = N_pp_g0.6 ... 1.2, 7-13 = S_pp_g0.6 ... 1.2.
Copy of `ros2-workspace-lab-dpc/src/dpc-controller/data/vehicle_1_lab_pp1014` (used there via `data/use_data.sh pp1014`).
Evaluation controller: `dpc-controller/config/controller/dpc/cpm_lab_controller_raceline_pp_eval1014.yaml` (dpc-controller
9d9d74f, deepc 16fe5f4 for the oracle labels); per-run results in `evaluation/04_hardware/hardware_runs.json`.
