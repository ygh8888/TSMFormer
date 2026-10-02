#!/bin/bash
# Briareo per-class 다중 시드 (R1-8 대응)
#   - 데이터셋 공식 val 분할 사용 → D안 재학습 불필요, 기존 백본 그대로
#   - late fusion이 99.31%로 포화되어 변별력이 낮으므로 per-class만 반복
set -u
cd /data/TSMFormer
mkdir -p logs_revision hyperparameters/_runtime

CFG="hyperparameters/Briareo/train_perclass.json"
TAG="bri_perclass"

echo "===== Briareo 시작 $(date '+%F %T') ====="
for S in 1994 2024 777 42 3407; do
  LOG="logs_revision/stage2_${TAG}_s${S}.log"
  CKPT="experiments/BL5_revision/fusion/${TAG}/seed${S}/checkpoints/Briareo/best_${TAG}_s${S}.pth"

  if [ -f "$CKPT" ] && grep -q "FINAL TEST" "$LOG" 2>/dev/null; then
    echo "[skip] seed=${S}"; continue
  fi

  RT=$(python3 make_seed_cfg_bri.py "$CFG" "$S" "$TAG") || { echo "[cfg fail] $S"; continue; }
  echo "########## ${TAG} seed=${S}  $(date '+%F %T')"
  python train_perclass.py --hypes "$RT" --seed "$S" > "$LOG" 2>&1
  echo "  rc=$? $(date '+%F %T')"
  grep -E "FINAL TEST|Best val" "$LOG" | tail -2
done
echo "===== Briareo 종료 $(date '+%F %T') ====="
