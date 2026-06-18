import numpy as np
import matplotlib.pyplot as plt
import gym
import yaml
from argparse import Namespace

# -----------------------------
# Load env and extract MB params
# -----------------------------
map_name = 'l_shape'
constant_friction = 1.1  # test at 0.5

with open('configs/config_%s.yaml' % map_name) as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

env = gym.make('f110_gym:f110-v0',
               map=conf.map_path,
               map_ext=conf.map_ext,
               num_agents=1,
               timestep=0.001,
               model='MB',
               drive_control_mode='acc',
               steering_control_mode='angle')

# set friction
env.params['tire_p_dy1'] = constant_friction * 0.9
env.params['tire_p_dx1'] = constant_friction

# extract and print ALL MB params for audit
mb_params = env.sim.agents[0].params
print("\n--- MB model parameters ---")
for k, v in mb_params.items():
    print(f"  {k}: {v}")

# extract the specific ones we care about
print("\n--- Key tire params ---")
print(f"  tire_p_cy1 (C): {mb_params['tire_p_cy1']}")
print(f"  tire_p_dy1 (mu_y): {mb_params['tire_p_dy1']:.4f}  (= {constant_friction} * 0.9)")
print(f"  tire_p_ey1 (E): {mb_params['tire_p_ey1']}")
print(f"  tire_p_ky1 (K/Fz): {mb_params['tire_p_ky1']}")
print(f"  tire_p_hy1 (S_hy): {mb_params['tire_p_hy1']}")
print(f"  tire_p_vy1 (S_vy/Fz): {mb_params['tire_p_vy1']}")
print(f"  m: {mb_params['m']}")
print(f"  lf: {mb_params['lf']}")
print(f"  lr: {mb_params['lr']}")
print(f"  I: {mb_params['I_z']}")

# compute derived NMPC params from MB params
m   = mb_params['m']
lf  = mb_params['lf']
lr  = mb_params['lr']
g   = 9.81
Fz_f = m * lr * g / (lf + lr)
Fz_r = m * lf * g / (lf + lr)

tire_p_dy1 = mb_params['tire_p_dy1']
tire_p_cy1 = mb_params['tire_p_cy1']
tire_p_ky1 = abs(mb_params['tire_p_ky1'])
tire_p_ey1 = mb_params['tire_p_ey1']

Bf = tire_p_ky1 / (tire_p_cy1 * tire_p_dy1)
Br = Bf
Cf = tire_p_cy1
Cr = tire_p_cy1
Df = tire_p_dy1 * Fz_f
Dr = tire_p_dy1 * Fz_r
Ef = tire_p_ey1
Er = tire_p_ey1

print(f"\n--- Derived NMPC params ---")
print(f"  Fz_f = {Fz_f:.1f} N")
print(f"  Fz_r = {Fz_r:.1f} N")
print(f"  Bf = Br = {Bf:.4f}")
print(f"  Cf = Cr = {Cf:.4f}")
print(f"  Df = {Df:.1f} N")
print(f"  Dr = {Dr:.1f} N")
print(f"  Ef = Er = {Ef:.7f}")

# -----------------------------
# Run both models with same inputs
# -----------------------------
raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
waypoints = np.array(raceline)
start_point = 1

# reset env to known state
obs, _, _, _ = env.reset(np.array([[
    waypoints[start_point, 1],
    waypoints[start_point, 2],
    waypoints[start_point, 3] + 1.5707963268,
    0.0, 8.0, 0.0, 0.0
]]))

# set friction again after reset
env.params['tire_p_dy1'] = constant_friction * 0.9
env.params['tire_p_dx1'] = constant_friction

# define a simple fixed input sequence: constant speed + step steer
# this gives clean, reproducible excitation
n_steps    = 100   # 10 seconds at dt=0.1
dt_control = 0.1
num_sim_steps = int(dt_control / env.timestep)  # 100 sim steps per control step

# fixed inputs: mild acceleration then step steer
inputs = []
for i in range(n_steps):
    acc   = 1.0 if i < 20 else 0.0          # accelerate for 2s then coast
    steer = 0.0 if i < 30 else 0.15         # step steer at t=3s
    inputs.append([acc, steer])

# run MB sim and log states
mb_states = []
x0_mb = np.array([
    env.sim.agents[0].state[3],   # vx
    env.sim.agents[0].state[10],  # vy
    env.sim.agents[0].state[5],   # yaw rate
])
mb_states.append(x0_mb)

for acc, steer in inputs:
    for _ in range(num_sim_steps):
        obs, _, _, _ = env.step(np.array([[steer, acc]]))
    mb_states.append(np.array([
        env.sim.agents[0].state[3],   # vx
        env.sim.agents[0].state[10],  # vy
        env.sim.agents[0].state[5],   # yaw rate
    ]))

mb_states = np.array(mb_states)

# -----------------------------
# Bicycle model RK4
# -----------------------------
ro  = 1.225
S   = 2.9
Cd  = 0.35

def bicycle_ode(x, u):
    vx, vy, r = x
    acc, delta = u

    # Exact NMPC formulas with fr0=fr1=fr4=0, ro=0, S=0, Cd=0
    ro = 0.0
    S  = 0.0
    Cd = 0.0
    fr0 = 0.0
    fr1 = 0.0
    fr4 = 0.0

    v_kmh = np.sqrt(vx**2 + vy**2) * 3.6
    fr    = fr0 + fr1 * v_kmh/100.0 + fr4 * (v_kmh/100.0)**4  # = 0.0

    Fz_f = m * lr * g / (lf + lr)
    Fz_r = m * lf * g / (lf + lr)
    Fr_f = fr * Fz_f   # = 0.0
    Fr_r = fr * Fz_r   # = 0.0

    Faero = 0.5 * ro * S * Cd * vx**2  # = 0.0

    F_braking = 0
    Fb_f = 2/3 * F_braking   # = 0.0
    Fb_r = 1/3 * F_braking   # = 0.0
    Fd   = m * acc

    Fx_f = -Fb_f - Fr_f      # = 0.0
    Fx_r = Fd - Fb_r - Fr_r  # = m * acc

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

    # banking = 0 so Fbanking_x = Fbanking_y = 0
    vx_dot = (Fx_r - Faero - Fy_f * np.sin(delta) + Fx_f * np.cos(delta) + m * vy * r) / m
    vy_dot = (Fy_r + Fy_f * np.cos(delta) + Fx_f * np.sin(delta) - m * vx * r) / m
    r_dot  = (lf * (Fy_f * np.cos(delta) + Fx_f * np.sin(delta)) - lr * Fy_r) / Iz

    return np.array([vx_dot, vy_dot, r_dot])


def rk4_step(x, u, dt, n_substeps=3):
    h = dt / n_substeps
    for _ in range(n_substeps):
        k1 = bicycle_ode(x,           u)
        k2 = bicycle_ode(x + h/2*k1, u)
        k3 = bicycle_ode(x + h/2*k2, u)
        k4 = bicycle_ode(x + h*k3,   u)
        x = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x


# run bicycle model from same x0
Iz = mb_params['I_z']
x_bic = mb_states[0].copy()
bic_states = [x_bic.copy()]
for u in inputs:
    x_bic = rk4_step(x_bic, u, dt_control)
    bic_states.append(x_bic.copy())
bic_states = np.array(bic_states)

# -----------------------------
# Plot comparison
# -----------------------------
t = np.arange(n_steps + 1) * dt_control
labels = ['vx [m/s]', 'vy [m/s]', 'yaw rate [rad/s]']

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle(f'MB vs Bicycle model — μ={constant_friction}', fontsize=13)

for j, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(t, mb_states[:, j],  label='MB (ground truth)', color='tab:blue')
    ax.plot(t, bic_states[:, j], label='Bicycle model',     color='tab:orange', linestyle='--')
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(f'mb_vs_bicycle_mu{constant_friction}.png', dpi=150)
plt.show()

# variants = {
#     'mu=0.5 correct (B=36, D=3412)': (36.0636, 36.0636, 3412.5, 1999.2),
#     'mu=0.5 D, mu=1.1 B (B=16, D=3412)': (16.3925, 16.3925, 3412.5, 1999.2),
#     'mu=1.1 correct (B=16, D=7507)': (16.3925, 16.3925, 7507.4, 4398.3),
# }
variants = {
    'mu=0.5 analytical (B=36, D=3412)': (36.0636, 36.0636, 3412.5, 1999.2),
    'mu=1.1 analytical (B=16, D=7507)': (16.3925, 16.3925, 7507.4, 4398.3),
    'mu=0.5 fitted (B=22.7, D=3043)':   (22.6863, 27.8174, 3043.4, 2384.8),  # NEW
    'mu=1.1 fitted (B=22.7, D=3043)':   (26.4965, 20.3698, 6046.9, 4372.1),  # NEW
}

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle('Variant comparison at μ=0.5 MB sim', fontsize=13)

for label, (Bf_v, Br_v, Df_v, Dr_v) in variants.items():
    Bf, Br, Df, Dr = Bf_v, Br_v, Df_v, Dr_v
    x_v = mb_states[0].copy()
    traj = [x_v.copy()]
    for u in inputs:
        x_v = rk4_step(x_v, u, dt_control)
        traj.append(x_v.copy())
    traj = np.array(traj)
    for j, ax in enumerate(axes):
        ax.plot(t, traj[:, j], label=label)

for j, (ax, lbl) in enumerate(zip(axes, labels)):
    ax.plot(t, mb_states[:, j], 'k-', linewidth=2, label='MB ground truth')
    ax.set_ylabel(lbl)
    ax.legend(fontsize=8)
    ax.grid(True)

axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig('variant_comparison.png', dpi=150)
plt.show()

print(f"\n--- Final state errors (t={n_steps*dt_control:.0f}s) ---")
for j, label in enumerate(labels):
    print(f"  {label:20s}  MB: {mb_states[-1,j]:.4f}   Bicycle: {bic_states[-1,j]:.4f}   Error: {abs(mb_states[-1,j]-bic_states[-1,j]):.4f}")