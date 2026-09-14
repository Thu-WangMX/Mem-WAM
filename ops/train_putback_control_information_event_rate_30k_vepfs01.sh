#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/control_information_v5_event_full_forced_half_putback_e2e_30k_seed42}"

exec bash /mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_event_conditioned_rate/ops/train_putback_control_information_event_rate_30k.sh
