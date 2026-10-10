#!/bin/bash
# Diagnostic (2026-10): closed-loop runs of sampling variants on l_shape, logs go to ../logs.
# Each variant runs the multi-friction track (11 laps) and uniform mu=0.5 / 1.1 (3 laps), in parallel.
#   bash evaluation/exploratory/segment_sampling/closed_loop/run.sh "TAG|ENV VARS|HYDRA OVERRIDES" ...
# Variants used in the README (SEG=controller.path_sampling.sampling=segment,++controller.path_sampling.segment_length):
#   "_slide40x6||$SEG=40,controller.num_traj=240"                  segment sampling 4 s x 6
#   "_islack||++controller.input_slack=true"                       input slack as in the lab
#   "_tini2|DPC_INIT_HORIZON=2|controller.initialization_horizon=2"  0.2 s window
#   "_unpaired|DPC_DATA_DIR=data/dpc_experiments/02_tracking_unpaired|"  unpaired data
#   "_noise|DPC_MEAS_NOISE=1|"                                     measurement noise
# A new window length or dataset is preprocessed on first use: run one variant alone first.
cd "$(dirname "$0")/../../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1  # small batched solves: threading only adds contention
LOGS=evaluation/exploratory/segment_sampling/logs
run() {
  local tag=$1 envs=$2 ov=$3 kind=$4 suffix
  if [ "$kind" = multi ]; then suffix=$tag; extra=""
  else suffix=_uniform_${kind}$tag; extra="DPC_USE_DYN_FRICTION=0 DPC_CONSTANT_FRICTION=$kind DPC_NUMBER_OF_LAPS=3"; fi
  env $envs $extra DPC_LOG_SUFFIX=$suffix DPC_HEADLESS=1 DPC_HYDRA_OVERRIDES=$ov .venv/bin/python DPC.py > $LOGS/stdout$suffix.txt 2>&1
  mv log_dpc_eval_l_shape$suffix $LOGS/ 2>/dev/null
  echo "$suffix: $(grep -h 'Real elapsed\|Error' $LOGS/stdout$suffix.txt | tail -1)"
}
for spec in "$@"; do
  IFS='|' read -r tag envs ov <<< "$spec"
  for kind in multi 0.5 1.1; do run "$tag" "$envs" "$ov" $kind & sleep 5; done
done; wait
