"""Prediction-based implied friction of the pooled DPC (evaluation only, does not change the control).

For the current step, three data-driven predictors with the controller's regularisation predict the outputs for the inputs
the controller just planned: from the controller's own (pooled) columns, and from columns sampled the same way from the
mu 0.5 data only and the mu 1.1 data only. The implied friction is the mix of the two single-dataset predictions that best
reproduces the pooled one, implied mu = w * 0.5 + (1 - w) * 1.1 ("the pooled DPC predicts like a car with friction mu").
Where the two single-dataset predictions hardly differ (distance), friction cannot be read off the prediction.
"""
import numpy as np


def ls_predict(Up, Uf, Yp, Yf, u_ini, y_ini, u_f, lambda_g, lambda_sigma, input_slack=False, lambda_sigma_u=None):
    """Regularised DeePC predictor with the future inputs fixed: min lg |g|^2 + ls |Yp g - y_ini|^2 (+ lsu |Up g - u_ini|^2)
    s.t. Uf g = u_f, sum g = 1 (and Up g = u_ini without input slack); returns Yf g."""
    n = Up.shape[1]
    H = lambda_g * np.eye(n) + lambda_sigma * Yp.T @ Yp
    f = lambda_sigma * Yp.T @ y_ini
    A, b = [Uf, np.ones((1, n))], [u_f, [1.0]]
    if input_slack:
        lsu = lambda_sigma_u or lambda_sigma
        H = H + lsu * Up.T @ Up
        f = f + lsu * Up.T @ u_ini
    else:
        A.insert(0, Up)
        b.insert(0, u_ini)
    A, b = np.vstack(A), np.concatenate(b)
    m = A.shape[0]
    K = np.block([[2 * H, A.T], [A, np.zeros((m, m))]])
    g = np.linalg.lstsq(K, np.concatenate([2 * f, b]), rcond=None)[0][:n]
    return Yf @ g


def implied_friction(planner_dpc, frictions=(0.5, 1.1), labels=(0, 1)):
    """(implied mu, distance [m] between the two single-dataset predictions (rms position over the horizon), relative fit
    residual, distance [m] of the pooled prediction to the mu-0.5 one, and to the mu-1.1 one) for the step the controller
    just solved; None values if the step has no solution."""
    dpc = planner_dpc.vehicle_dpc.dpc
    cfg, c, data = planner_dpc.cfg, planner_dpc.cfg.controller, planner_dpc.data
    # VehicleDPC.plan denormalises/unaligns the solution arrays in place after the solve, so the window, reference and
    # planned inputs are taken from the copies the DPC logs inside solve() (normalised, aligned frame of this step).
    buf = getattr(planner_dpc.experiment_logger, "_dpc_buf", None)
    if not buf or not buf.get("future_inputs") or buf["future_inputs"][-1] is None:
        return None, None, None, None, None
    u_ini, y_ini, ref, u_f = (np.asarray(buf[k][-1], float).flatten(order="F")
                              for k in ("initial_inputs", "initial_outputs", "reference_trajectory", "future_inputs"))
    args = dict(lambda_g=float(c.lambda_g), lambda_sigma=float(c.lambda_sigma), input_slack=bool(c.get("input_slack", False)),
                lambda_sigma_u=c.get("lambda_sigma_u", None))
    pred = {"pooled": ls_predict(dpc.input_matrix_past.value, dpc.input_matrix_future.value, dpc.output_matrix_past.value,
                                 dpc.output_matrix_future.value, u_ini, y_ini, u_f, **args)}
    saved = getattr(data, "oracle_label", None)
    try:
        for key, label in zip(("low", "high"), labels):
            data.oracle_label = label
            Up, Uf, Yp, Yf, _ = data.select_trajectories(cfg, initial_inputs=u_ini, initial_outputs=y_ini, reference_future=ref)
            pred[key] = ls_predict(Up, Uf, Yp, Yf, u_ini, y_ini, u_f, **args)
    finally:
        data.oracle_label = saved
    P, A, B = pred["pooled"], pred["low"], pred["high"]
    D = A - B
    w = float(np.clip((P - B) @ D / max(D @ D, 1e-12), 0.0, 1.0))
    fit = float(np.linalg.norm(P - (w * A + (1 - w) * B)) / max(np.linalg.norm(P - B), 1e-12))
    sd = np.asarray(planner_dpc.vehicle_dpc.output_std, float).ravel()
    n_out = len(sd)
    def pos_dist(X):                                 # rms position distance over the horizon [m] (normalised, aligned)
        Xm = X.reshape(-1, n_out).T
        return float(np.sqrt(np.mean((Xm[0] * sd[0]) ** 2 + (Xm[1] * sd[1]) ** 2)))
    return w * frictions[0] + (1 - w) * frictions[1], pos_dist(D), fit, pos_dist(P - A), pos_dist(P - B)
