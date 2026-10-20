# Simulation DPC data for the prediction study (Fig. 1), random excitation (2026-10-18)

Random inputs on the full-scale Sao Paulo track at constant friction mu 0.5 / 1.1, 1000 s each at 10 Hz (10000 rows), recorded
with the data-collection experiment of external/dpc (f1tenth multibody backend, friction applied as tire_p_dx1 = mu,
tire_p_dy1 = 0.9 mu, as in the multi-friction simulation):

    cd external/dpc
    PYTHONPATH=.:../..:../nmpc python experiment/main.py +experiment=vehicle_collect_data experiment.friction.mu=<mu> \
        seed=31415 experiment.duration_sec=1000 hydra.run.dir=<output folder>

The random policy (experiment.random_policy in the stored .hydra/config.yaml) redraws steering and speed targets every 100
steps. Same protocol as the earlier mu 0.3 / 0.7 / 1.0 data in ../01_prediction (branch archive/full-2026-10), at the friction values of the closed-loop
evaluation. The logger's time column `t` is coarse (it advances every 100 rows); the dataset loader does not use it.
