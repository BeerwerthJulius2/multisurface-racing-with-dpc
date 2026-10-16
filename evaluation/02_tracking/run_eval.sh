#!/usr/bin/env bash
# Tracking evaluation of the multisurface DPC (pooled / oracle / inverse oracle) and the NMPC baseline, as run on 2026-10-16.
# usage: evaluation/02_tracking/run_eval.sh [map] [seeds] [arms]   (from the repository root; default: l_shape "1 2 3"
#        "pooled oracle inverse"; the paper uses l_shape with all arms and Nuerburgring with pooled). RUN_NMPC=0: DPC only.
# l_shape runs on the friction-aware speed profile (friction_aware_profile.py: same share of the grip in every friction zone).
#
# Environment: python 3.8 venv in .venv (gym 0.19, f1tenth_gym multibody branch, numpy 1.22, cvxpy 1.5, osqp 1.1, hydra 1.3,
# casadi 3.6.7, torch cpu), acados v0.5.4 built in ~/dev/acados (ACADOS_SOURCE_DIR, LD_LIBRARY_PATH below).
# Data: data/dpc_experiments/02_tracking_mu_aware (see its README). Logs: evaluation/02_tracking/results/
# log_dpc_eval_<map>_eval_<arm>_s<seed>, log_nmpc_eval_<map>_s<seed> (overwrites the committed results; compare with git diff).
set -ef   # -f: no globbing of the bracketed override values
MAP=${1:-l_shape}
SEEDS=${2:-"1 2 3"}
ARMS=${3-"pooled oracle inverse"}
LAPS=11; [ "$MAP" = Nuerburgring ] && LAPS=4   # the first lap is discarded
OUT=evaluation/02_tracking/results
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD/external/dpc:$PWD/external/nmpc
export ACADOS_SOURCE_DIR=${ACADOS_SOURCE_DIR:-$HOME/dev/acados} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-$HOME/dev/acados/lib}
COMMON="DPC_MAP_NAME=$MAP DPC_NUMBER_OF_LAPS=$LAPS DPC_VELOCITY_MAX=20 DPC_ABORT_ERROR=5"
[ "$MAP" = l_shape ] && COMMON="$COMMON DPC_WPT_PATH=./maps/l_shape/waypoints/l_shape_mintime_l720_mu_aware.csv"
# DPC setting tuned for pooled tracking (lambda_g 5, lambda_sigma 50, speed weight 300, reference spaced by the profile speed);
# kept for the friction-aware profile (a smaller lambda_g tracks better there but predicts worse)
Q="[[10.1,0,0,0],[0,10.1,0,0],[0,0,300,0],[0,0,0,20.1]]"
DPC="DPC_HEADLESS=1 DPC_DATA_DIR=data/dpc_experiments/02_tracking_mu_aware DPC_REF_SPACING=profile"
DPC="$DPC DPC_HYDRA_OVERRIDES=controller.lambda_g=5;controller.lambda_sigma=50;controller.q_k=$Q;controller.q_fk=$Q"
mkdir -p $OUT
for s in $SEEDS; do
  for a in $ARMS; do
    case $a in pooled) O="DPC_ORACLE=0 DPC_IMPLIED_FRICTION=1";; oracle) O="DPC_ORACLE=1";; inverse) O="DPC_ORACLE=inverse";; esac
    env $COMMON $DPC $O DPC_SEED=$s DPC_LOG_SUFFIX=_eval_${a}_s$s .venv/bin/python DPC.py
    mv log_dpc_eval_${MAP}_eval_${a}_s$s $OUT/
  done
  [ "${RUN_NMPC:-1}" = 1 ] || continue   # NMPC: one run at a time (acados generates its solver in the working directory)
  env $COMMON DPC_SEED=$s DPC_LOG_SUFFIX=_s$s .venv/bin/python evaluation/02_tracking/run_nmpc_headless.py
  mv log_nmpc_eval_${MAP}_s$s $OUT/
done
