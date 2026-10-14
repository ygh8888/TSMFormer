#!/bin/bash
# LoRA (color, IR) + bottleneck, aligned sampler + full adapted-backbone state saving
set -u
cd "$(dirname "$0")"
mkdir -p logs_revision hyperparameters/_runtime
TAG=lora_p0_fix2
CFG=hyperparameters/NVGestures/train_cmaf_v4lora.json
echo "===== $TAG 시작 $(date '+%F %T') ====="
for S in 1994 2024 777 42 3407; do
  LOG=logs_revision/stage2_${TAG}_s${S}.log
  if grep -q "FINAL TEST" "$LOG" 2>/dev/null; then echo "[skip] seed=$S"; continue; fi
  RT=$(python3 make_seed_cfg.py "$CFG" "$S" "$TAG") || { echo "[cfg fail] $S"; continue; }
  echo "########## $TAG seed=$S  $(date '+%F %T')"
  python train_cmaf_v4_lora.py --hypes "$RT" --seed "$S" > "$LOG" 2>&1
  echo "  rc=$?  $(date '+%F %T')"
  grep -E "Full backbone|VAL CHECK|FINAL TEST" "$LOG"
done
echo "===== $TAG 종료 $(date '+%F %T') ====="
