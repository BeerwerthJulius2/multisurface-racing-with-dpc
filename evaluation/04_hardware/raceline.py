"""Closed raceline shared by pure_pursuit_node.py and dpc_controller_node.py (reference_source raceline), so data
collection and the DPC follow the identical line and speed profile.

The raceline file (make_raceline.py; ';'-separated, '#' header lines) holds s; x; y; yaw; kappa; v. The speed profile
can be recomputed on load for a chosen grip level a_lat (the friction): v = min(v_max, sqrt(a_lat / |kappa|)) with
acceleration and braking limits on the friction ellipse (the longitudinal acceleration available shrinks with the
lateral acceleration used), around the closed loop.
"""
import math

import numpy as np


def speed_profile(s, kappa, a_lat, v_max, a_acc, a_dec, friction_ellipse=True):
    """Maximum speed along the closed path s (arc length) with curvature kappa."""
    k_abs = np.maximum(np.abs(kappa), 1e-6)
    v = np.minimum(v_max, np.sqrt(a_lat / k_abs))
    ds = np.diff(s)

    def step(v_from, k, a_max, d):
        if friction_ellipse:
            a_max = a_max * math.sqrt(1.0 - min((v_from ** 2 * k / a_lat) ** 2, 1.0))
        return math.sqrt(v_from ** 2 + 2 * a_max * d)

    n = len(v)
    for _ in range(3):                                         # closed loop: repeat so the limits wrap around
        for k in range(1, n + 1):                              # forward pass: acceleration
            v[k % n] = min(v[k % n], step(v[k - 1], k_abs[k - 1], a_acc, ds[(k - 1) % (n - 1)]))
        for k in range(n - 2, -2, -1):                         # backward pass: braking
            v[k % n] = min(v[k % n], step(v[(k + 1) % n], k_abs[(k + 1) % n], a_dec, ds[k % (n - 1)]))
    return v


def load_raceline(path, a_lat=None, v_max=1.2, a_acc=0.4, a_dec=0.6, friction_ellipse=True):
    """Raceline dict (s, x, y, yaw, kappa, v, length). With a_lat (> 0) the speed profile is recomputed for that grip
    level; otherwise the profile stored in the file is used."""
    R = np.loadtxt(path, delimiter=";", comments="#")
    line = {"s": R[:, 0], "x": R[:, 1], "y": R[:, 2], "yaw": R[:, 3], "kappa": R[:, 4], "v": R[:, 5].copy(),
            "length": float(R[-1, 0] + (R[1, 0] - R[0, 0]))}
    if a_lat is not None and a_lat > 0:
        line["v"] = speed_profile(line["s"], line["kappa"], a_lat, v_max, a_acc, a_dec, friction_ellipse)
    return line


def project(line, x, y, last_index=None, window=150):
    """Arc length of the nearest raceline point, the segment index and the distance [m]. Searched near last_index when
    given (the figure-8 passes through the central crossing twice)."""
    n = len(line["s"])
    idx = np.arange(n) if last_index is None else (last_index + np.arange(-20, window)) % n
    xs, ys = line["x"][idx], line["y"][idx]
    d = np.c_[line["x"][(idx + 1) % n] - xs, line["y"][(idx + 1) % n] - ys]
    t = np.clip(((x - xs) * d[:, 0] + (y - ys) * d[:, 1]) / np.maximum((d ** 2).sum(1), 1e-12), 0.0, 1.0)
    dist = np.hypot(xs + t * d[:, 0] - x, ys + t * d[:, 1] - y)
    k = int(np.argmin(dist))
    i = int(idx[k])
    ds = line["s"][1] - line["s"][0]
    return float((line["s"][i] + t[k] * ds) % line["length"]), i, float(dist[k])


def sample(line, s, key):
    """Raceline value (x, y, v, kappa) at arc length(s) s, closed loop."""
    L = line["length"]
    return np.interp(np.mod(s, L), np.r_[line["s"], L], np.r_[line[key], line[key][0]])


def sample_yaw(line, s):
    """Heading at arc length(s) s, interpolated across the +-pi boundary."""
    n = len(line["s"])
    ds = line["s"][1] - line["s"][0]
    s = np.mod(np.atleast_1d(s), line["length"])
    k = np.floor(s / ds).astype(int) % n
    r = (s - line["s"][k]) / ds
    y0, y1 = line["yaw"][k], line["yaw"][(k + 1) % n]
    return np.angle(np.exp(1j * (y0 + r * np.angle(np.exp(1j * (y1 - y0))))))
