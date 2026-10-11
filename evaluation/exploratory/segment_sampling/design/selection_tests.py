"""Diagnostic (2026-10): what makes a selection pick the matching friction? Small offline tests.

Data: the controller's normalized dataset; queries are recorded windows in corners, their time neighbourhood
removed from both recordings. Labels are used only for scoring.

Test 1: a segment's window-fit residual mixes how close the segment passes to the current operating point
with whether its dynamics match. AUC = probability that a matching segment has a smaller residual than a
non-matching one; overall vs. among segments that pass equally close (distance within +-10%).

Test 2: upper bound with labels: fit the window with each dataset's n nearest columns (affine ridge fit)
and pick the dataset with the smaller residual. How well can the window tell the friction at all?

Test 3: purity vs. tracking. For a pool of columns: share from the matching friction, and how well the
pool can realize the recorded future as reference while staying consistent with the past (the QP's
constraints and penalties, future inputs free, Q = I). Two-stage selection: the window admits the top-K
segments (fit, ridge 0.01), the reference picks the columns within them (contextual distance).

Usage:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python evaluation/exploratory/segment_sampling/design/selection_tests.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
sys.path.insert(0, os.path.join(REPO_ROOT, "external", "dpc"))
from dataset.data_sampling.samplers import segment_precompute  # noqa: E402

DATASET = "data/dpc_experiments/02_tracking/pure_pursuit_0_5/preprocessed/preprocessed_2026-03-23T15-52-48.npy"
RAW = ["data/dpc_experiments/02_tracking/pure_pursuit_0_5/inputs_outputs_0.5.npy",
       "data/dpc_experiments/02_tracking/pure_pursuit_1_1/inputs_outputs_1.1.npy"]  # labels 0, 1
T_INI, NU, NY = 5, 2, 4
EXCLUDE = 30
N_QUERY = 120  # corner queries per recording


# lab-like conditions: 0.2 s window (T_ini = 2), and mu=0.5 recorded at another speed (unpaired)
SHORT_WINDOW = ("data/dpc_experiments/02_tracking/pure_pursuit_0_5/preprocessed/preprocessed_2026-10-11T00-01-53.npy", RAW, 2)
UNPAIRED = ("data/dpc_experiments/02_tracking_unpaired/pure_pursuit_0_5_v08/preprocessed/preprocessed_2026-10-11T00-05-27.npy",
            ["data/dpc_experiments/02_tracking_unpaired/pure_pursuit_0_5_v08/inputs_outputs_0.5.npy", RAW[1]], T_INI)


def setup(dataset=DATASET, raw=RAW, t_ini=T_INI, full=False):
    d = np.load(os.path.join(REPO_ROOT, dataset), allow_pickle=True).item()
    U, Y, labels = np.asarray(d["input_matrix"]), np.asarray(d["output_matrix"]), np.asarray(d["dataset_labels"])
    Up, Yp = U[:t_ini * NU], Y[:t_ini * NY]
    start = {lab: np.flatnonzero(labels == lab)[0] for lab in np.unique(labels)}
    index_in_run = np.arange(len(labels)) - np.array([start[l] for l in labels])
    lat = np.empty(len(labels))
    for lab, path in enumerate(raw):
        a = np.load(os.path.join(REPO_ROOT, path))
        la = np.abs(a[:, 6] * np.gradient(np.unwrap(a[:, 7]), 0.1))
        cols = labels == lab
        lat[cols] = la[index_in_run[cols] + t_ini - 1]
    corner = lat > np.percentile(lat, 80)
    rng = np.random.default_rng(0)
    pool = corner if not full else np.ones_like(corner)
    queries = np.concatenate([rng.choice(np.flatnonzero((labels == lab) & pool & (index_in_run > EXCLUDE)
                                                        & (index_in_run < (labels == lab).sum() - EXCLUDE)),
                                         N_QUERY, replace=False) for lab in (0, 1)])
    if full:
        return Up, Yp, labels, index_in_run, queries, U[t_ini * NU:], Y[t_ini * NY:], corner, lat
    return Up, Yp, labels, index_in_run, queries


def auc(score_match, score_other):
    """P(matching score < other score), from ranks (ties count half)."""
    s = np.concatenate([score_match, score_other])
    ranks = np.argsort(np.argsort(s)) + 1.0
    r = ranks[:len(score_match)].sum()
    n1, n2 = len(score_match), len(score_other)
    return 1 - (r - n1 * (n1 + 1) / 2) / (n1 * n2)


def test1(Up, Yp, labels, index_in_run, queries, G=25, ridge=0.1):
    cache = segment_precompute(Up, Yp, NU, segment_length=G, ridge=ridge)
    starts = cache["starts"]
    H = np.vstack([Up, Yp])
    out = {0: [], 1: []}
    for q in queries:
        w = H[:, q]
        excluded = np.abs(index_in_run - index_in_run[q]) <= EXCLUDE
        blocked = np.concatenate([[0], np.cumsum(excluded)])
        valid = blocked[starts + G] == blocked[starts]
        s = starts[valid]
        r = np.linalg.norm(cache["M"][valid] @ w + cache["b"][valid], axis=1)
        dist = np.linalg.norm(H - w[:, None], axis=0)
        d = np.lib.stride_tricks.sliding_window_view(dist, G).min(axis=1)[s]  # closest pass of the segment
        match = labels[s] == labels[q]
        def topk(score):
            """share matching among the best k NON-overlapping segments, as the sampler takes them"""
            used, picked = np.zeros(len(labels), bool), []
            for i in np.argsort(score):
                if not used[s[i]:s[i] + G].any():
                    used[s[i]:s[i] + G] = True
                    picked.append(match[i])
                    if len(picked) == 12:
                        break
            return [np.mean(picked[:k]) for k in (1, 4, 12)]
        out[labels[q]].append(topk(r) + topk(d))
    print(f"Test 1 (G={G}, ridge={ridge}): share of matching segments among the top k, corner queries (chance ~0.5)")
    print(f"{'query mu':>9} {'by residual k=1/4/12':>22} {'by distance k=1/4/12':>22}")
    for lab, mu in ((0, 0.5), (1, 1.1)):
        a = np.mean(out[lab], axis=0)
        print(f"{mu:>9} {a[0]:>8.2f}{a[1]:>6.2f}{a[2]:>6.2f}   {a[3]:>8.2f}{a[4]:>6.2f}{a[5]:>6.2f}")

def affine_fit_residual(A, w, ridge):
    m = A.shape[1]
    K = np.block([[A.T @ A + ridge * np.eye(m), np.ones((m, 1))], [np.ones((1, m)), np.zeros((1, 1))]])
    g = np.linalg.solve(K, np.r_[A.T @ w, 1.0])[:m]
    return np.linalg.norm(A @ g - w)


def test2(Up, Yp, labels, index_in_run, queries, ns=(5, 10, 20, 40, 80), ridges=(0.01, 0.1, 1.0)):
    H = np.vstack([Up, Yp])
    print("Test 2: correct dataset decision in corners, labelled group fit (rows n, columns ridge)")
    print(f"{'n':>4} " + "  ".join(f"{'ridge ' + str(rg) + ' mu=0.5/1.1':>22}" for rg in ridges))
    res = {(n, rg): {0: [], 1: []} for n in ns for rg in ridges}
    for q in queries:
        w = H[:, q]
        keep = np.abs(index_in_run - index_in_run[q]) > EXCLUDE
        dist = np.linalg.norm(H - w[:, None], axis=0)
        for n in ns:
            near = {k: np.flatnonzero(keep & (labels == k))[np.argsort(dist[keep & (labels == k)])[:n]] for k in (0, 1)}
            for rg in ridges:
                r = [affine_fit_residual(H[:, near[k]], w, rg) for k in (0, 1)]
                res[(n, rg)][labels[q]].append(int(np.argmin(r)) == labels[q])
    for n in ns:
        print(f"{n:>4} " + "  ".join(f"{np.mean(res[(n, rg)][0]):>15.0%} / {np.mean(res[(n, rg)][1]):.0%}" for rg in ridges))


def track_residual(Up, Yp, Yf, u, y, y_ref, lambda_g=1.1, lambda_sigma=10.1):
    """QP proxy: min lambda_g |g|^2 + lambda_sigma |Yp g - y|^2 + |Yf g - y_ref|^2 s.t. Up g = u, 1^T g = 1."""
    n = Up.shape[1]
    Q = lambda_g * np.eye(n) + lambda_sigma * Yp.T @ Yp + Yf.T @ Yf
    C = np.vstack([Up, np.ones((1, n))])
    K = np.block([[2 * Q, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
    rhs = np.concatenate([2 * (lambda_sigma * Yp.T @ y + Yf.T @ y_ref), u, [1.0]])
    g = np.linalg.lstsq(K, rhs, rcond=None)[0][:n]
    return np.sqrt(np.mean((Yf @ g - y_ref) ** 2))


def test3(Up, Yp, labels, index_in_run, queries, Uf, Yf, corner, lat, n_cols=100):
    H = np.vstack([Up, Yp])
    caches = {(G, rg): segment_precompute(Up, Yp, NU, segment_length=G, ridge=rg) for G, rg in [(25, 0.1), (25, 0.01)]}

    def segments(q, G, rg, k):
        c = caches[(G, rg)]
        excluded = np.abs(index_in_run - index_in_run[q]) <= EXCLUDE
        blocked = np.concatenate([[0], np.cumsum(excluded)])
        valid = blocked[c["starts"] + G] == blocked[c["starts"]]
        r = np.linalg.norm(c["M"][valid] @ H[:, q] + c["b"][valid], axis=1)
        used, sel = np.zeros(len(labels), bool), []
        for j in c["starts"][valid][np.argsort(r)]:
            if not used[j:j + G].any():
                used[j:j + G] = True
                sel.append(np.arange(j, j + G))
                if len(sel) == k:
                    break
        return np.concatenate(sel)

    def contextual(q, candidates, n):
        """Deterministic contextual score: z(past distance) + z(future distance to the reference, inputs to 0)."""
        past = np.linalg.norm(H[:, candidates] - H[:, [q]], axis=0)
        fut = np.linalg.norm(Yf[:, candidates] - Yf[:, [q]], axis=0) + np.linalg.norm(Uf[:, candidates], axis=0)
        z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
        return candidates[np.argsort(z(past) + z(fut))[:n]]

    def extend(sel, G, e, keep):
        """Each selected segment grown by e columns on both sides along its own recording (same pass)."""
        cols = set()
        for a in sel[::G]:
            for c in range(a - e, a + G + e):
                if 0 <= c < len(labels) and labels[c] == labels[a] and keep[c]:
                    cols.add(c)
        return np.array(sorted(cols))

    def run(q, keep, n=40, rg=0.01):
        """Run (recording) whose n nearest columns fit the window best: assumes one condition per run."""
        dist = np.linalg.norm(H - H[:, [q]], axis=0)
        r = []
        for k in (0, 1):
            cand = np.flatnonzero(keep & (labels == k))
            r.append(affine_fit_residual(H[:, cand[np.argsort(dist[cand])[:n]]], H[:, q], rg))
        return int(np.argmin(r))

    methods = {
        "contextual, top 100": lambda q, keep: contextual(q, np.flatnonzero(keep), n_cols),
        "contextual, top 300": lambda q, keep: contextual(q, np.flatnonzero(keep), 300),
        "segments 25x4, ridge 0.1": lambda q, keep: segments(q, 25, 0.1, 4),
        "segments 25x4, ridge 0.01": lambda q, keep: segments(q, 25, 0.01, 4),
        "segments 25x12, ridge 0.01": lambda q, keep: segments(q, 25, 0.01, 12),
        "segments 25x4, extended to 50": lambda q, keep: extend(segments(q, 25, 0.01, 4), 25, 12, keep),
        "segments 25x4, extended to 75": lambda q, keep: extend(segments(q, 25, 0.01, 4), 25, 25, keep),
        "segments 25x6, extended to 50": lambda q, keep: extend(segments(q, 25, 0.01, 6), 25, 12, keep),
        "run decision -> contextual 100": lambda q, keep: contextual(q, np.flatnonzero(keep & (labels == run(q, keep))), n_cols),
        "run decision -> contextual 300": lambda q, keep: contextual(q, np.flatnonzero(keep & (labels == run(q, keep))), 300),
        "2-stage: 12 segments -> 100": lambda q, keep: contextual(q, segments(q, 25, 0.01, 12), n_cols),
        "2-stage: 24 segments -> 100": lambda q, keep: contextual(q, segments(q, 25, 0.01, 24), n_cols),
    }
    res = {m: [] for m in methods}
    for q in queries:
        keep = np.abs(index_in_run - index_in_run[q]) > EXCLUDE
        for m, f in methods.items():
            sel = f(q, keep)
            res[m].append((np.mean(labels[sel] == labels[q]),
                           track_residual(Up[:, sel], Yp[:, sel], Yf[:, sel], Up[:, q], Yp[:, q], Yf[:, q]), corner[q]))
    print(f"Test 3: pool purity and reference tracking residual (normalized units), {len(queries)} windows")
    print(f"{'method':<30} {'all: purity':>11} {'track':>7} | {'corners: purity':>15} {'track':>7}")
    for m, r in res.items():
        r = np.array(r)
        c = r[:, 2].astype(bool)
        print(f"{m:<30} {r[:, 0].mean():>11.0%} {np.median(r[:, 1]):>7.3f} | {r[c, 0].mean():>15.0%} {np.median(r[c, 1]):>7.3f}")


if __name__ == "__main__":
    if "3" in sys.argv[1:]:
        test3(*setup(full=True))
        sys.exit()
    if "lab" in sys.argv[1:]:
        for name, cond in (("short window", SHORT_WINDOW), ("unpaired", UNPAIRED)):
            print(f"== {name}")
            data = setup(*cond)
            test2(*data, ns=(20, 40, 80), ridges=(0.01,))
            for G in (25, 40):
                test1(*data, G=G, ridge=0.01)
        sys.exit()
    data = setup()
    if "2" in sys.argv[1:]:
        test2(*data)
    else:
        for G, ridge in [(25, 0.1), (25, 0.01), (40, 0.01)]:
            test1(*data, G=G, ridge=ridge)
