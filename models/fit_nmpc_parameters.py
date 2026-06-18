import numpy as np
from scipy.optimize import least_squares

# -----------------------------
# vehicle parameters (global)
# -----------------------------
m  = 1225.887
Iz = 1538.853371
lf = 0.88392
lr = 1.50876
ro = 0.0
S  = 0.0
Cd = 0.0
g  = 9.81

Fz_f = m * lr * g / (lf + lr)  # ≈ 7587 N
Fz_r = m * lf * g / (lf + lr)  # ≈ 4446 N


# -----------------------------
# NMPC dynamics (one-step)
# -----------------------------
def f_ode(x, u, theta):
    vx, vy, r = x
    acc, delta = u

    # All four params free — from theta
    Bf, Br, Df, Dr = theta

    # Fixed from MB ground truth
    Cf = 1.3507
    Cr = 1.3507
    Ef = -0.0074722
    Er = -0.0074722

    Fx_f  = 0.0
    Fx_r  = m * acc
    Faero = 0.0

    if vx > 1e-3:
        alpha_f = delta - np.arctan((vy + lf * r) / vx)
        alpha_r = np.arctan((lr * r - vy) / vx)
    else:
        alpha_f = 0.0
        alpha_r = 0.0

    Fy_f_lat = Df * np.sin(Cf * np.arctan(Bf * alpha_f - Ef * (Bf * alpha_f - np.arctan(Bf * alpha_f))))
    Fy_r_lat = Dr * np.sin(Cr * np.arctan(Br * alpha_r - Er * (Br * alpha_r - np.arctan(Br * alpha_r))))

    Fmax_f = np.sqrt(Fz_f**2 + (Cf * Fz_f)**2)
    Fmax_r = np.sqrt(Fz_r**2 + (Cr * Fz_r)**2)
    Gy_f = np.clip(Fx_f / Fmax_f, -0.98, 0.98)
    Gy_r = np.clip(Fx_r / Fmax_r, -0.98, 0.98)
    Fy_f = Fy_f_lat * np.cos(np.arcsin(Gy_f))
    Fy_r = Fy_r_lat * np.cos(np.arcsin(Gy_r))

    vx_dot = (Fx_r - Faero - Fy_f * np.sin(delta) + Fx_f * np.cos(delta) + m * vy * r) / m
    vy_dot = (Fy_r + Fy_f * np.cos(delta) + Fx_f * np.sin(delta) - m * vx * r) / m
    r_dot  = (lf * (Fy_f * np.cos(delta) + Fx_f * np.sin(delta)) - lr * Fy_r) / Iz

    return np.array([vx_dot, vy_dot, r_dot])


def f(x, u, theta, dt, n_steps=3):
    """RK4 with n_steps substeps — matches acados ERK4 x3 exactly"""
    h = dt / n_steps
    for _ in range(n_steps):
        k1 = f_ode(x,           u, theta)
        k2 = f_ode(x + h/2*k1, u, theta)
        k3 = f_ode(x + h/2*k2, u, theta)
        k4 = f_ode(x + h*k3,   u, theta)
        x = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x


# -----------------------------
# finite-horizon residuals
# -----------------------------
def residuals(theta, X, U, segment_id, dt, H):
    res = []
    for k in range(len(U) - H):
        # Skip if window crosses a segment boundary
        if not np.all(segment_id[k:k + H + 1] == segment_id[k]):
            continue

        x_pred = X[k].copy()
        for h in range(H):
            x_pred = f(x_pred, U[k + h], theta, dt)
            res.append(x_pred - X[k + h + 1])

    if len(res) == 0:
        return np.zeros(3)
    return np.concatenate(res)


# -----------------------------
# load data
# -----------------------------
data_11 = np.load("data/nmpc_experiments/nmpc_data_1.1.npy")
data_05 = np.load("data/nmpc_experiments/nmpc_data_0.5.npy")

data = np.vstack([data_11, data_05])
# data = data_11
# data = data_05

X = np.stack([data[:, 7], data[:, 8], data[:, 9]], axis=1)  # vx, vy, r
U = np.stack([data[:, 2], data[:, 3]], axis=1)               # acc, delta
segment_id = data[:, 1]  # col1 is now segment ID

dt = 0.1
H  = 10

# -----------------------------
# initial guess — theta = [Bf, Br, mu_f, mu_r]
# -----------------------------
# theta = [Bf, Br, Df, Dr] — all four free
theta0 = np.array([
    (22.69 + 26.50) / 2,   # Bf ≈ 24.6
    (27.82 + 20.37) / 2,   # Br ≈ 24.1
    (3043 + 6047) / 2,     # Df ≈ 4545
    (2385 + 4372) / 2,     # Dr ≈ 3379
])

lower = np.array([ 1.0,  1.0,   500.0,   500.0])
upper = np.array([50.0, 50.0, 15000.0, 15000.0])  # widen Dr upper bound

res = least_squares(residuals, theta0, bounds=(lower, upper),
                    args=(X, U, segment_id, dt, H),
                    method='trf', verbose=2)

Bf_opt, Br_opt, Df_opt, Dr_opt = res.x

print("\n--- Fitted parameters ---")
print(f"  Bf   = {Bf_opt:.4f}")
print(f"  Br   = {Br_opt:.4f}")
print(f"  Df   = {Df_opt:.1f} N")
print(f"  Dr   = {Dr_opt:.1f} N")
print(f"  Cost:       {res.cost:.6f}")
print(f"  Optimality: {res.optimality:.2e}")
print(f"  Message:    {res.message}")