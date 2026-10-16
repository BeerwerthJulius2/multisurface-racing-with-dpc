# multisurface-racing

## DPC under varying friction (paper results)
Data, raw result logs, figures and Table I of "Data-Driven Predictive Control for Autonomous Racing under Varying Friction"
(Beerwerth, Tischbein, Alrifaee) are committed; the scripts below regenerate them.

Setup (Python 3.8):
```
git submodule update --init external/dpc external/nmpc        # paper/ is the Overleaf project (needs an Overleaf token)
python3.8 -m venv .venv && .venv/bin/pip install "pip<24.1" && .venv/bin/pip install -r evaluation/requirements.txt
.venv/bin/pip install -e <f1tenth_gym, branch multibody, e55925e> -e <acados v0.5.4>/interfaces/acados_template
```
acados is built in `~/dev/acados` (else set `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`). All commands from the repository root.

| Paper | Data | Results | Run / plot |
|---|---|---|---|
| Fig. 1 prediction | `data/dpc_experiments/01_prediction` | `evaluation/01_prediction/prediction_results.json` | `DPC_prediction_error.py`; `evaluation/01_prediction/visualize_prediction_errors.py <results.json>` |
| Figs. 2-3, Table I (L-shape) | `data/dpc_experiments/02_tracking_v2` (README) | `evaluation/02_tracking/results` | `evaluation/02_tracking/run_eval.sh`; `visualization_dpc.py l_shape`; `evaluation/02_tracking/table_tracking.py` |
| Fig. 4 (Nürburgring) | same | same | `run_eval.sh Nuerburgring "1 2 3" pooled`; `visualization_dpc.py Nuerburgring` |
| Hardware figure | `data/dpc_experiments/04_hardware` (README) | `evaluation/04_hardware/hardware_runs.json` | lab runs (ros2-workspace-lab-dpc); `extract_lab_runs.py <workspace>`; `evaluation/04_hardware/plot_hardware.py` |

NMPC baseline: Pacejka parameters in `NMPC.py` (paper Sec. IV-B; fitting script `models/fit_nmpc_parameters.py`, data
`data/nmpc_experiments`). Figure style (colours, sizes): `helpers/paper_style.py`.

## Simulator
This repository is using multi-body full scale car version of the [F1/10 gym](https://github.com/atomyks/f1tenth_gym/tree/multibody).

## Entry points
* GP_MPC.py
    * Runs the first lap using a standard controller (Extended-Kinematic MPC or Pure Pursuit) and collects data for GP training.
    * Trains the GP using the collected dataset and then switches to GP MPC.
    * After every lap, it saves the dataset generated from driving on the track to a file called "testing_dataset.json".
* GP_MPC_eval.py
    * Loads in the dataset created by GP_MPC.py and retrains the GP on this dataset.
    * The trained GP(s) are then evaluated for prediction error.
* Kinematic_and_dynamic_MPC.py.py
    * You can choose from extended-kinematic MPC, dynamic (bicycle model) MPC, and pure pursuit controler.
    * Currently used for comparison against GP-MPC.

## Citing
If you find this Controller useful, please consider citing:

```
@article{nagy2023ensemble,
  title={Ensemble Gaussian Processes for Adaptive Autonomous Driving on Multi-friction Surfaces},
  author={Nagy, Tom{\'a}{\v{s}} and Amine, Ahmad and Nghiem, Truong X and Rosolia, Ugo and Zang, Zirui and Mangharam, Rahul},
  journal={arXiv preprint arXiv:2303.13694},
  year={2023}
}
```
