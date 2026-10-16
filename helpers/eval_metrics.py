"""Evaluation measures shared by the paper figures and Table I."""
import numpy as np


def prediction_error(pred_xy, x, y):
    """rms position error over the horizon of each logged prediction; the first predicted point belongs to the logged step
    that fits it best (the logger's offset)."""
    horizon = next(len(p[0]) for p in pred_xy if p is not None)
    best = None
    for off in (-1, 0, 1):
        e0 = [np.hypot(p[0][0] - x[k + off], p[1][0] - y[k + off]) for k, p in enumerate(pred_xy) if p is not None and 0 <= k + off < len(x)]
        if best is None or np.median(e0) < best[1]:
            best = (off, np.median(e0))
    err = np.full(len(x), np.nan)
    for k, p in enumerate(pred_xy):
        j = k + best[0]
        if p is not None and j >= 0 and j + horizon <= len(x):
            err[k] = np.sqrt(np.mean((np.asarray(p[0]) - x[j:j + horizon]) ** 2 + (np.asarray(p[1]) - y[j:j + horizon]) ** 2))
    return err
