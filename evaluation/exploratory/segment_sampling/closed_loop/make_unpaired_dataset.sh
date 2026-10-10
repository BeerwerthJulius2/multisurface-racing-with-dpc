#!/bin/bash
# Diagnostic (2026-10): record the unpaired mu=0.5 dataset (data/dpc_experiments/02_tracking_unpaired).
# Pure pursuit on l_shape at a constant 0.8x speed profile, 7 laps, no rendering. The original recordings
# ramp the speed lap by lap identically for both frictions (paired); this one overlaps the mu=1.1 data
# only around one speed level. The mu=1.1 half is a symlink to the original recording.
cd "$(dirname "$0")/../../../.."
OUT=data/dpc_experiments/02_tracking_unpaired/pure_pursuit_0_5_v08
sed -e "s/^    map_name = 'Nuerburgring'/    map_name = 'l_shape'/" \
    -e "s/^    use_dyn_friction = True/    use_dyn_friction = False/" \
    -e "s/^    constant_friction = 0.7/    constant_friction = 0.5/" \
    -e "s/^    velocity_profile_multiplier = 1.0/    velocity_profile_multiplier = 0.8/" \
    -e "s/^    number_of_laps = 1/    number_of_laps = 7/" \
    -e "s/^    env.render()/    pass/" -e "s/env.render(mode='human_fast')/pass/" \
    -e "s|f'data/dpc_experiments/inputs_outputs_{model_to_use}_{constant_friction}|f'$OUT/inputs_outputs_{constant_friction}|" \
    DPC_collect_data.py > /tmp/collect_unpaired.py
mkdir -p $OUT && PYTHONPATH=$PWD .venv/bin/python /tmp/collect_unpaired.py && rm -f $OUT/inputs_outputs_0.5.csv
# the loader also needs a .hydra/config.yaml per run folder (untracked, as for the original data):
# copy it from data/dpc_experiments/02_tracking/pure_pursuit_0_5/.hydra
