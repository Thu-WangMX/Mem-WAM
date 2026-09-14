#!/usr/bin/env bash
set -euo pipefail

# Restart the event-conditioned experiment from the official FullKV initialization.
# A new output directory is mandatory: the v5 run has only a weight-only 5k
# checkpoint, so resuming it would reset optimizer/scheduler/global-step state.
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/control_information_v6_event_full_forced_half_putback_e2e_30k_seed42}"

exec bash /mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_event_conditioned_rate/ops/train_putback_control_information_event_rate_30k.sh
