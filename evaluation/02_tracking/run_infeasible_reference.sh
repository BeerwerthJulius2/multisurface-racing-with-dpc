#!/usr/bin/env bash
# Infeasible-reference check (Section IV-D): constant friction, the speed profile scaled by 1.0 and 1.1, DPC with the pooled /
# matched (oracle) / mismatched (inverse) data, 2 laps, runs end when the car leaves the track (DPC_ABORT_ERROR). At mu 0.5 the
# 1.1x profile exceeds the grip; at mu 1.1 it is feasible. Same controller setting as run_eval.sh. Logs:
# evaluation/02_tracking/results/infeasible_reference/; numbers: python evaluation/02_tracking/infeasible_reference.py
set -ef
OUT=evaluation/02_tracking/results/infeasible_reference
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD/external/dpc:$PWD/external/nmpc
DPC="DPC_HEADLESS=1 DPC_DATA_DIR=data/dpc_experiments/02_tracking_v2 DPC_REF_SPACING=profile"
DPC="$DPC DPC_HYDRA_OVERRIDES=controller=multisurface_sim"   # lambda_g 5, lambda_sigma 50, speed weight 300
mkdir -p $OUT
for mu in 0.5 1.1; do for v in 1.0 1.1; do for a in pooled oracle inverse; do
  [ "$mu" = 1.1 ] && [ "$v" = 1.0 ] && continue
  case $a in pooled) O="DPC_ORACLE=0 DPC_IMPLIED_FRICTION=1";; oracle) O="DPC_ORACLE=1";; inverse) O="DPC_ORACLE=inverse";; esac
  tag=_inf_mu${mu}_v${v}_${a}
  env DPC_MAP_NAME=l_shape DPC_NUMBER_OF_LAPS=2 DPC_VELOCITY_MAX=20 DPC_ABORT_ERROR=5 DPC_MAX_TIME=200 DPC_USE_DYN_FRICTION=0 \
      DPC_CONSTANT_FRICTION=$mu DPC_VELOCITY_MULTIPLIER=$v $DPC $O DPC_SEED=1 DPC_LOG_SUFFIX=$tag .venv/bin/python DPC.py
  mv log_dpc_eval_l_shape$tag $OUT/
done; done; done
