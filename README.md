# multisurface-racing

## DPC under varying friction (paper results)
Data, raw result logs, figures and tables of "Data-Driven Predictive Control for Autonomous Racing under Varying Friction"
(Beerwerth, Tischbein, Alrifaee) are committed; the scripts below regenerate them. Earlier datasets and exploratory
diagnostics are on branch `archive/full-2026-10`.

Setup (Python 3.8):
```
git submodule update --init external/dpc external/nmpc
python3.8 -m venv .venv && .venv/bin/pip install "pip<24.1" && .venv/bin/pip install -r evaluation/requirements.txt
.venv/bin/pip install -e <f1tenth_gym, branch multibody, e55925e> -e <acados v0.5.4>/interfaces/acados_template
```
acados is built in `~/dev/acados` (else set `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`). All commands from the repository root.

| Paper | Data | Results | Run / plot |
|---|---|---|---|
| Fig. 1 prediction | `data/dpc_experiments/01_prediction_v2` (README) | `evaluation/01_prediction/prediction_results_random.json` (T_ini 10), `..._tini5.json` (T_ini 5) | `evaluation/01_prediction/prediction_error_eval.py` (settings in the header of `visualize_prediction_errors.py`, `PRED_TINI`); `visualize_prediction_errors.py <results.json>`; grip use of the datasets: `grip_usage.py` |
| Fig. 2 tracking, Fig. 3 implied friction, Table I | `data/dpc_experiments/02_tracking_v2` (README) | `evaluation/02_tracking/results` | `evaluation/02_tracking/run_eval.sh`; `visualization_dpc.py l_shape`; `evaluation/02_tracking/table_tracking.py` |
| Nürburgring (Table I) | same | same | `run_eval.sh Nuerburgring "1 2 3" pooled` |
| Infeasible reference (Sec. IV-D) | same | `evaluation/02_tracking/results/infeasible_reference` | `evaluation/02_tracking/run_infeasible_reference.sh`; `infeasible_reference.py` |
| Fig. 4, Table II hardware | `data/dpc_experiments/04_hardware` (README; data, run configs and logs) | `evaluation/04_hardware/hardware_runs.json` | `evaluation/04_hardware/extract_lab_runs.py`; `evaluation/04_hardware/plot_hardware.py` |

"matched" / "mismatched" = the oracle / inverse runs (DPC_ORACLE=1 / inverse: sampling restricted to the dataset of the true /
the other friction). NMPC baseline: Pacejka parameters and g-g limits in `external/nmpc/Config/RoboRacer-Multibody` (midpoint
of the mu 0.5 and 1.1 fits; excitation data from `NMPC_collect_data.py` in `data/nmpc_experiments`, fit with
`models/fit_nmpc_parameters.py`). Numbers for the text: `evaluation/PAPER_NOTES.md`; figure style: `helpers/paper_style.py`.

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
* DPC.py: DPC on a track with spatially varying friction (data in `data/dpc_experiments`, controller in `external/dpc`).
* NMPC.py: nominal NMPC baseline (`external/nmpc`).
* DPC_collect_data.py, NMPC_collect_data.py: data collection with pure pursuit or MPC at constant friction.
* visualization_dpc.py: paper figures from the evaluation logs.

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
