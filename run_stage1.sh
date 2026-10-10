#!/bin/bash
set -u; cd "$(dirname "$0")"; mkdir -p logs_revision
for m in depth ir normal optflow; do
  echo "##### $m $(date '+%F %T')"
  python main.py --hypes hyperparameters/NVGestures_rev/train_${m}_rev.json --phase train \
    > logs_revision/stage1_${m}.log 2>&1
  echo "  exit=$? $(date '+%F %T')"
done
echo "===== 1단계 완료 $(date '+%F %T') ====="
ls -la experiments/BL5_revision/NVGestures/checkpoints/NVGestures/
