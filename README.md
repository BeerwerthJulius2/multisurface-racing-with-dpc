# multisurface-racing

## DPC under varying friction (paper results)
Data, raw result logs, figures and tables of "Data-Driven Predictive Control for Autonomous Racing under Varying Friction"
(Beerwerth, Tischbein, Alrifaee) are committed; the scripts below regenerate them. Earlier datasets and exploratory
diagnostics are on branch `archive/full-2026-10`.

Setup (Python 3.8):
```
git submodule update --init external/dpc external/nmpc external/f1tenth_gym
python3.8 -m venv .venv && .venv/bin/pip install "pip<24.1" && .venv/bin/pip install -r evaluation/requirements.txt
.venv/bin/pip install -e external/f1tenth_gym -e <acados v0.5.4>/interfaces/acados_template
```
acados is built in `~/dev/acados` (else set `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`). All commands from the repository root.
The runs are deterministic: on the machine that produced the committed logs they reproduce bit for bit. On other hardware,
floating-point differences of about 1e-7 at the first solve grow along the closed loop (millimetres to a few centimetres); the
statements of the paper are unaffected.

| Paper | Data | Results | Run / plot |
|---|---|---|---|
| Fig. 1 prediction | `data/dpc_experiments/01_prediction_v2` (README) | `evaluation/01_prediction/prediction_results_random.json` (T_ini 10), `..._tini5.json` (T_ini 5) | `evaluation/01_prediction/prediction_error_eval.py` (settings in the header of `visualize_prediction_errors.py`; T_ini 5: the same with `PRED_TINI=5 PRED_OUT=evaluation/01_prediction/prediction_results_random_tini5.json`); `visualize_prediction_errors.py <results.json>`; grip use of the datasets: `grip_usage.py` |
| Fig. 2 tracking, Fig. 3 implied friction, Table I | `data/dpc_experiments/02_tracking_v2` (README) | `evaluation/02_tracking/results` | `evaluation/02_tracking/run_eval.sh`; `visualization_dpc.py l_shape`; `evaluation/02_tracking/table_tracking.py` |
| Nürburgring (Table I) | same | same | `run_eval.sh Nuerburgring "1 2 3" pooled` |
| Uncapped Nürburgring (Sec. IV-D) | same | `evaluation/PAPER_NOTES.md` | `VMAX=100 RUN_NMPC=0 run_eval.sh Nuerburgring 1 pooled` (stops at the first solve) |
| Infeasible reference (supplementary, not in the paper) | same | `evaluation/02_tracking/results/infeasible_reference` | `evaluation/02_tracking/run_infeasible_reference.sh`; `infeasible_reference.py` |
| Fig. 4, Table II hardware | `data/dpc_experiments/04_hardware` (README; data, run configs and logs) | `evaluation/04_hardware/hardware_runs.json` | `evaluation/04_hardware/extract_lab_runs.py`; `evaluation/04_hardware/plot_hardware.py` |

"matched" / "mismatched" = the oracle / inverse runs (DPC_ORACLE=1 / inverse: sampling restricted to the dataset of the true /
the other friction). NMPC baseline: Pacejka parameters and g-g limits in `external/nmpc/Config/RoboRacer-Multibody` (midpoint
of the mu 0.5 and 1.1 fits; excitation data from `NMPC_collect_data.py` in `data/nmpc_experiments`, fit with
`models/fit_nmpc_parameters.py`). Numbers for the text: `evaluation/PAPER_NOTES.md`; figure style: `helpers/paper_style.py`.

## Simulator
This repository is using multi-body full scale car version of the [F1/10 gym](https://github.com/atomyks/f1tenth_gym/tree/multibody)
(submodule `external/f1tenth_gym`, branch multibody, e55925e).

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
If you use the DPC code or data, please cite:

```
@unpublished{beerwerth2026datadriven,
  title={Data-Driven Predictive Control for Autonomous Racing under Varying Friction},
  author={Beerwerth, Julius and Tischbein, Julius and Alrifaee, Bassam},
  note={Manuscript in preparation},
  year={2026}
}
```

If you find this Controller useful, please consider citing:

```
@article{nagy2023ensemble,
  title={Ensemble Gaussian Processes for Adaptive Autonomous Driving on Multi-friction Surfaces},
  author={Nagy, Tom{\'a}{\v{s}} and Amine, Ahmad and Nghiem, Truong X and Rosolia, Ugo and Zang, Zirui and Mangharam, Rahul},
  journal={arXiv preprint arXiv:2303.13694},
  year={2023}
}
```
