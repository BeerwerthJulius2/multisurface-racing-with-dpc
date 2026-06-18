import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# -----------------------------
# Vehicle parameters (matching MB model)
# -----------------------------
m   = 1225.887
Iz  = 1538.853371
lf  = 0.88392
lr  = 1.50876
ro  = 0.0
S   = 0.0
Cd  = 0.0
g   = 9.81

Fz_f = m * lr * g / (lf + lr)
Fz_r = m * lf * g / (lf + lr)

print(f"Fz_f = {Fz_f:.1f} N")
print(f"Fz_r = {Fz_r:.1f} N")

# -----------------------------
# Pacejka params derived from MB ground truth
# -----------------------------
tire_p_cy1    =  1.3507
tire_p_ey1    = -0.0074722
tire_p_ky1    =  21.92      # abs value
tire_p_hy1    =  0.0026747
tire_p_vy1    =  0.037318

def get_pacejka_params(constant_friction):
    tire_p_dy1 = constant_friction * 0.9
    Cf = tire_p_cy1
    Cr = tire_p_cy1
    Ef = tire_p_ey1
    Er = tire_p_ey1
    Df = tire_p_dy1 * Fz_f
    Dr = tire_p_dy1 * Fz_r
    Bf = tire_p_ky1 / (tire_p_cy1 * tire_p_dy1)
    Br = tire_p_ky1 / (tire_p_cy1 * tire_p_dy1)
    S_vy_f = Fz_f * tire_p_vy1
    S_vy_r = Fz_r * tire_p_vy1
    S_hy    = tire_p_hy1
    return Bf, Br, Cf, Cr, Df, Dr, Ef, Er, S_hy, S_vy_f, S_vy_r

# -----------------------------
# Bicycle model ODE (NMPC formulation)
# -----------------------------
def bicycle_ode(x, u, friction, include_shifts=False):
    vx, vy, r = x
    acc, delta = u

    Bf, Br, Cf, Cr, Df, Dr, Ef, Er, S_hy, S_vy_f, S_vy_r = get_pacejka_params(friction)

    v_kmh = np.sqrt(vx**2 + vy**2) * 3.6
    fr = 0.0 + 0.0 * v_kmh/100.0 + 0.0 * (v_kmh/100.0)**4

    Fr_f = fr * Fz_f
    Fr_r = fr * Fz_r
    Fx_f = -Fr_f
    Fx_r = m * acc - Fr_r
    Faero = 0.5 * ro * S * Cd * vx**2

    if vx > 1e-3:
        alpha_f = delta - np.arctan((vy + lf * r) / vx)
        alpha_r = np.arctan((lr * r - vy) / vx)
    else:
        alpha_f = 0.0
        alpha_r = 0.0

    # optionally include horizontal shift (S_hy)
    if include_shifts:
        alpha_f_eff = alpha_f + S_hy
        alpha_r_eff = alpha_r + S_hy
    else:
        alpha_f_eff = alpha_f
        alpha_r_eff = alpha_r

    Fy_f_lat = Df * np.sin(Cf * np.arctan(Bf * alpha_f_eff - Ef * (Bf * alpha_f_eff - np.arctan(Bf * alpha_f_eff))))
    Fy_r_lat = Dr * np.sin(Cr * np.arctan(Br * alpha_r_eff - Er * (Br * alpha_r_eff - np.arctan(Br * alpha_r_eff))))

    # optionally include vertical shift (S_vy)
    if include_shifts:
        Fy_f_lat += S_vy_f
        Fy_r_lat += S_vy_r

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


def rk4_step(x, u, friction, dt, n_steps=3, include_shifts=False):
    h = dt / n_steps
    for _ in range(n_steps):
        k1 = bicycle_ode(x,           u, friction, include_shifts)
        k2 = bicycle_ode(x + h/2*k1, u, friction, include_shifts)
        k3 = bicycle_ode(x + h/2*k2, u, friction, include_shifts)
        k4 = bicycle_ode(x + h*k3,   u, friction, include_shifts)
        x = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x


# -----------------------------
# Load MB data (ground truth)
# -----------------------------
for friction_label, filename in [('1.1', 'data/dpc_experiments/nmpc_data_1.1.npy'),
                                  ('0.5', 'data/dpc_experiments/nmpc_data_0.5.npy')]:

    data = np.load(filename)
    X_mb = np.stack([data[:, 7], data[:, 8], data[:, 9]], axis=1)  # vx, vy, r
    U    = np.stack([data[:, 2], data[:, 3]], axis=1)               # acc, delta
    friction = float(friction_label)
    dt = 0.1
    H  = 10

    # -----------------------------
    # Open-loop rollout comparison
    # -----------------------------
    stride = 50  # evaluate every 50 timesteps
    indices = range(0, len(U) - H, stride)

    err_bicycle        = np.zeros((len(indices), H, 3))
    err_bicycle_shifts = np.zeros((len(indices), H, 3))

    for i, k in enumerate(indices):
        x_bic        = X_mb[k].copy()
        x_bic_shifts = X_mb[k].copy()

        for h in range(H):
            x_bic        = rk4_step(x_bic,        U[k+h], friction, dt, include_shifts=False)
            x_bic_shifts = rk4_step(x_bic_shifts, U[k+h], friction, dt, include_shifts=True)

            err_bicycle[i, h]        = x_bic        - X_mb[k+h+1]
            err_bicycle_shifts[i, h] = x_bic_shifts - X_mb[k+h+1]

    t_axis   = (np.arange(H) + 1) * dt
    state_labels = ['vx [m/s]', 'vy [m/s]', 'r [rad/s]']

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    fig.suptitle(f'Bicycle model vs MB ground truth  —  μ={friction_label}', fontsize=13)

    for j, (ax, label) in enumerate(zip(axes, state_labels)):
        mae_bic    = np.abs(err_bicycle[:, :, j]).mean(axis=0)
        mae_shifts = np.abs(err_bicycle_shifts[:, :, j]).mean(axis=0)
        std_bic    = np.abs(err_bicycle[:, :, j]).std(axis=0)
        std_shifts = np.abs(err_bicycle_shifts[:, :, j]).std(axis=0)

        ax.plot(t_axis, mae_bic,    label='Bicycle (no shifts)',   color='tab:blue')
        ax.plot(t_axis, mae_shifts, label='Bicycle (with S_hy/S_vy)', color='tab:orange', linestyle='--')
        ax.fill_between(t_axis, mae_bic - std_bic, mae_bic + std_bic, alpha=0.15, color='tab:blue')
        ax.fill_between(t_axis, mae_shifts - std_shifts, mae_shifts + std_shifts, alpha=0.15, color='tab:orange')
        ax.set_ylabel(f'MAE  {label}')
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Prediction horizon [s]')
    plt.tight_layout()
    plt.savefig(f'model_comparison_mu{friction_label}.png', dpi=150)
    plt.show()

    print(f"\n--- μ={friction_label} error at H={H} steps ---")
    for j, label in enumerate(state_labels):
        print(f"  {label:12s}  no shifts: {np.abs(err_bicycle[:,-1,j]).mean():.5f}"
              f"   with shifts: {np.abs(err_bicycle_shifts[:,-1,j]).mean():.5f}")