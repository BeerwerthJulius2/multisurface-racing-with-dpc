"""Friction-aware speed profile on the l_shape raceline: the minimum-time profile of the raceline is laid out for mu 0.5
(a_lat <= 4.0 m/s^2, i.e. at the mu 0.5 grip limit everywhere), so on the mu 0.8 / 1.1 zones the car stays far below their
limit. Here the speed is scaled by sqrt(mu / 0.5) (same share of the available grip in every zone), followed by a backward
(braking, a_x scaled by mu / 0.5 up to 3 m/s^2, the car's brake limit is 3.5) and forward (acceleration, unchanged: power limited) pass on a g-g ellipse so that the car slows down before
it enters a lower-friction zone. Same geometry; columns s, x, y, psi, kappa as in the raceline.
usage (repo root): friction_aware_profile.py <out.csv> [constant mu]   (no mu: the track's friction map)"""
import json, sys
import numpy as np
out = sys.argv[1]; mu_const = float(sys.argv[2]) if len(sys.argv) > 2 else None
src = "maps/l_shape/waypoints/l_shape_mintime_l720.csv"
header = open(src).read().splitlines()[:3]
wp = np.loadtxt(src, delimiter=";", skiprows=3)
if mu_const is None:   # friction at each raceline point from the friction map (as in DPC.py: nearest map point)
    tpamap = np.loadtxt("maps/l_shape/friction_data/l_shape_l720_track_tpamap.csv", delimiter=";", skiprows=1)
    tpadata = json.load(open("maps/l_shape/friction_data/l_shape_l720_track_tpadata.json"))
    idx = [int(np.argmin(np.hypot(tpamap[:, 0] - x, tpamap[:, 1] - y))) for x, y in wp[:, 1:3]]
    mu = np.array([tpadata[str(i)][0] for i in idx], float)
else:
    mu = np.full(len(wp), mu_const)
s, v0 = wp[:, 0], wp[:, 5]
ds = np.diff(np.r_[s, s[-1] + (s[-1] - s[-2])])            # closed loop: last point to the first
a_brake = np.minimum(4.21 * mu / 0.5, 3.0)   # the raceline profile's limit at mu 0.5; the car brakes with at most a_max 3.5
a_lat, a_acc = 4.0 * mu / 0.5, 2.12           # raceline profile at mu 0.5; acceleration power limited
kappa = np.abs(wp[:, 4])
ellipse = lambda vv, i: np.sqrt(max(0.0, 1 - (vv ** 2 * kappa[i] / a_lat[i]) ** 2))   # share of a_x left next to a_y (g-g ellipse)
v = v0 * np.sqrt(mu / 0.5)
for _ in range(3):                                          # passes over the closed loop
    for i in range(len(v) - 1, -1, -1):
        j = (i + 1) % len(v); v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_brake[i] * ellipse(v[j], i) * ds[i]))
    for i in range(len(v)):
        j = (i + 1) % len(v); v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_acc * ellipse(v[i], i) * ds[i]))
ax = (np.roll(v, -1) ** 2 - v ** 2) / (2 * ds)
w = wp.copy(); w[:, 5], w[:, 6] = v, ax
with open(out, "w") as f:
    f.write("\n".join(header) + "\n")
    np.savetxt(f, w, delimiter=";", fmt="%.7f")
al = v ** 2 * np.abs(wp[:, 4])
for z in np.unique(mu):
    m = mu == z; print(f"mu {z}: v {v[m].min():.1f}-{v[m].max():.1f} m/s (raceline profile {v0[m].min():.1f}-{v0[m].max():.1f}), "
                       f"a_lat max {al[m].max():.2f} m/s^2 = {al[m].max() / (0.9 * z * 9.81):.2f} of the lateral grip")
print("written", out)
