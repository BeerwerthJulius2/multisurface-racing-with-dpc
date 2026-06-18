'''
Test 1 — Self-prediction. Train on μ=0.3 predict on μ=0.3, same for 0.7 and 1.0.
Test 2 — Cross-prediction. Train on μ=1.0, predict on μ=0.3 (and vice versa).
Test 3 — Mixed data, no sampling. Pool {0.3, 0.7, 1.0}, predict on each.
Test 4 — Mixed data, with sampling. Same pool, same eval targets.
'''
import json
from copy import copy
from pathlib import Path

import hydra
import numpy as np
from hydra import initialize, compose
from hydra.utils import get_class
from matplotlib import pyplot as plt

from external.dpc.dataset.trajectory_dataset import TrajectoryDataset
from external.dpc.predictor.dpc_predictor import DPCPredictor


class EvaluatePredictionError:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):

        n_data = 500
        n_val = round(n_data*self.cfg.experiment.validation_split)
        n_train = n_data - n_val
        #n_val = 5
        print(f"n_data: {n_data}, n_val: {n_val}")
        t_ini_in = self.cfg.controller.initialization_horizon * self.cfg.controller.num_inputs
        t_ini_out = self.cfg.controller.initialization_horizon * self.cfg.controller.num_outputs

        # --- Load datasets ---
        datasets = {}
        for name in ["mu_03", "mu_07", "mu_10"]:
            cfg_mu = copy(self.cfg)
            cfg_mu.dataset.path = getattr(self.cfg.experiment.datasets, name)
            datasets[name] = TrajectoryDataset(cfg_mu)

        # --- Split train/val ---
        validation_input_matrices = {}
        validation_output_matrices = {}
        for name, dataset in datasets.items():
            dataset.input_matrix = dataset.input_matrix[:, :n_data]
            dataset.output_matrix = dataset.output_matrix[:, :n_data]
            n_cols = dataset.input_matrix.shape[1]
            train_idx, val_idx = self.split_columns(n_cols, self.cfg.experiment.validation_split, self.cfg.experiment.seed)
            validation_input_matrices[name] = dataset.input_matrix[:, val_idx[:n_val]]
            validation_output_matrices[name] = dataset.output_matrix[:, val_idx[:n_val]]
            dataset.input_matrix = dataset.input_matrix[:, train_idx[:n_train]]
            dataset.output_matrix = dataset.output_matrix[:, train_idx[:n_train]]

        # --- Build predictors ---
        # Test 1 & 2: one predictor per mu
        predictors = {name: DPCPredictor(ds) for name, ds in datasets.items()}

        # Test 3 & 4: pooled predictor
        pooled_in = np.hstack([ds.input_matrix for ds in datasets.values()])
        pooled_out = np.hstack([ds.output_matrix for ds in datasets.values()])
        pooled_predictor = DPCPredictor.from_matrices(datasets["mu_03"], pooled_in, pooled_out)
        random_predictor = DPCPredictor.from_matrices(datasets["mu_03"], pooled_in, pooled_out, cfg=copy(self.cfg))
        random_predictor.enable_sampling("random")
        sampled_predictor = DPCPredictor.from_matrices(datasets["mu_03"], pooled_in, pooled_out, cfg=copy(self.cfg))
        sampled_predictor.enable_sampling("contextual")

        # --- Evaluate ---
        def eval_predictor(predictor, eval_name, label):
            errors = []
            for i in range(n_val):
                print(f"\r{label}: {i+1}/{n_val}", end="", flush=True)
                initial_inputs = validation_input_matrices[eval_name][:t_ini_in, i]
                initial_outputs = validation_output_matrices[eval_name][:t_ini_out, i]
                future_inputs = validation_input_matrices[eval_name][t_ini_in:, i]
                future_outputs = validation_output_matrices[eval_name][t_ini_out:, i]

                pred = predictor.predict(
                    initial_inputs,
                    initial_outputs,
                    future_inputs,
                    #future_outputs,
                    )

                errors.append(np.sqrt(np.mean((pred - future_outputs) ** 2)))

                #fig, ax = plt.subplots()
                #plot_dpc(ax, initial_outputs=initial_outputs, predicted_outputs=pred,
                #         reference_outputs=future_outputs, output_matrix_past=predictor.Yp.value, output_matrix_future=predictor.Yf.value)
                #plt.show()
            print()

            return errors

        # Test 1: Self-prediction
        results_1 = {name: eval_predictor(predictors[name], name, f"T1 {name}") for name in datasets}

        # Test 2: Cross-prediction
        results_2 = {}
        for train, eval_ in [("mu_10", "mu_03"), ("mu_03", "mu_10")]:
            results_2[f"{train}→{eval_}"] = eval_predictor(predictors[train], eval_, f"T2 {train}→{eval_}")

        # Test 3: Mixed, no sampling
        results_3 = {name: eval_predictor(pooled_predictor, name, f"T3 {name}") for name in datasets}

        # Test 4: Mixed, random sampling
        results_4 = {name: eval_predictor(random_predictor, name, f"T4 {name}") for name in datasets}

        # Test 5: Mixed, contextual sampling
        results_5 = {name: eval_predictor(sampled_predictor, name, f"T5 {name}") for name in datasets}

        results = {
            "test_1_self": results_1,
            "test_2_cross": results_2,
            "test_3_mixed_full": results_3,
            "test_4_mixed_random": results_4,
            "test_5_mixed_contextual": results_5,
            "config": {
                "n_data": n_data,
                "n_train": n_train,
                "n_val": n_val,
                "num_traj": self.cfg.controller.num_traj,
                "seed": self.cfg.experiment.seed,
            },
        }

        Path("evaluation/01_prediction").mkdir(parents=True, exist_ok=True)

        with open("evaluation/01_prediction/prediction_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Results saved to prediction_results.json")

    def split_columns(self, n_cols, val_ratio, seed):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_cols)
        n_val = max(1, int(round(n_cols * val_ratio)))
        return indices[n_val:], indices[:n_val]


def main() -> None:
    with initialize(config_path="external/dpc/experiment/config", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=evaluate_prediction_error",
            ],
        )

    exp = EvaluatePredictionError(cfg)
    exp.run()

if __name__ == "__main__":
    main()