#!/usr/bin/env bash
# Tracking evaluation of the multisurface DPC (pooled / oracle / inverse oracle) and the NMPC baseline, as run on 2026-10-15.
# usage: evaluation/02_tracking/run_eval.sh [map] [seeds]     (from the repository root; default: l_shape "1 2 3")
#
# Environment: python 3.8 venv in .venv (gym 0.19, f1tenth_gym multibody branch, numpy 1.22, cvxpy 1.5, osqp 1.1, hydra 1.3,
# casadi 3.6.7, torch cpu), acados v0.5.4 built in ~/dev/acados (ACADOS_SOURCE_DIR, LD_LIBRARY_PATH below).
# Rendering: set DPC_HEADLESS=1 (DPC.py); NMPC.py always renders, run it through a wrapper that disables
# F110Env.render / add_render_callback for batch runs.
# Data: data/dpc_experiments/02_tracking_v2 (see its README). Logs: log_dpc_eval_<map>_<arm>_s<seed>, log_nmpc_eval_<map>_s<seed>.
set -ef   # -f: no globbing of the bracketed override values
MAP=${1:-l_shape}
SEEDS=${2:-"1 2 3"}
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD/external/dpc:$PWD/external/nmpc
export ACADOS_SOURCE_DIR=${ACADOS_SOURCE_DIR:-$HOME/dev/acados} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-$HOME/dev/acados/lib}
COMMON="DPC_MAP_NAME=$MAP DPC_NUMBER_OF_LAPS=11 DPC_VELOCITY_MAX=20 DPC_ABORT_ERROR=5"   # 11 laps, the first is discarded
# DPC setting tuned for pooled tracking (lambda_g 5, lambda_sigma 50, speed weight 300, reference spaced by the profile speed)
Q="[[10.1,0,0,0],[0,10.1,0,0],[0,0,300,0],[0,0,0,20.1]]"
DPC="DPC_HEADLESS=1 DPC_DATA_DIR=data/dpc_experiments/02_tracking_v2 DPC_REF_SPACING=profile"
DPC="$DPC DPC_HYDRA_OVERRIDES=controller.lambda_g=5;controller.lambda_sigma=50;controller.q_k=$Q;controller.q_fk=$Q"
for s in $SEEDS; do
  env $COMMON $DPC DPC_SEED=$s DPC_IMPLIED_FRICTION=1 DPC_ORACLE=0       DPC_LOG_SUFFIX=_pooled_s$s  .venv/bin/python DPC.py
  env $COMMON $DPC DPC_SEED=$s                         DPC_ORACLE=1       DPC_LOG_SUFFIX=_oracle_s$s  .venv/bin/python DPC.py
  env $COMMON $DPC DPC_SEED=$s                         DPC_ORACLE=inverse DPC_LOG_SUFFIX=_inverse_s$s .venv/bin/python DPC.py
  env $COMMON DPC_SEED=$s DPC_LOG_SUFFIX=_s$s .venv/bin/python NMPC.py
done
