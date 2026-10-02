#!/bin/bash
# D안 2단계: fusion 변형 다중 시드 재학습 (R1-1, R1-2, R1-5)
set -u
cd /data/TSMFormer
mkdir -p logs_revision hyperparameters/_runtime

SEEDS5="1994 2024 777 42 3407"
SEEDS3="1994 2024 777"

RUNS=(
  "perclass|hyperparameters/NVGestures/train_perclass.json|train_perclass.py|$SEEDS5"
  "lora_p0|hyperparameters/NVGestures/train_cmaf_v4lora.json|train_cmaf_v4_lora.py|$SEEDS5"
  "mbt_bl2in|hyperparameters/NVGestures/train_cmaf_v4_bl2in.json|train_cmaf_v4.py|$SEEDS5"
  "mbt_gate|hyperparameters/NVGestures/train_cmaf_v4.json|train_cmaf_v4.py|$SEEDS3"
  "dense|hyperparameters/NVGestures/train_cmaf.json|train_cmaf.py|$SEEDS3"
)

echo "===== 2단계 시작 $(date '+%F %T') ====="
TOTAL=0; DONE=0; FAIL=0

for entry in "${RUNS[@]}"; do
  IFS='|' read -r TAG CFG SCRIPT SEEDS <<< "$entry"
  for S in $SEEDS; do
    TOTAL=$((TOTAL+1))
    CKPT="experiments/BL5_revision/fusion/${TAG}/seed${S}/checkpoints/NVGestures/best_${TAG}_s${S}.pth"
    LOG="logs_revision/stage2_${TAG}_s${S}.log"

    # 체크포인트 + 로그의 FINAL TEST 둘 다 있어야 완료로 간주
    if [ -f "$CKPT" ] && grep -q "FINAL TEST" "$LOG" 2>/dev/null; then
      echo "[skip] ${TAG} seed=${S}"
      DONE=$((DONE+1)); continue
    fi

    RT=$(python3 make_seed_cfg.py "$CFG" "$S" "$TAG") || { echo "[cfg fail] $TAG $S"; FAIL=$((FAIL+1)); continue; }

    echo "########## ${TAG} seed=${S}  $(date '+%F %T')"
    python "$SCRIPT" --hypes "$RT" --seed "$S" > "$LOG" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
      DONE=$((DONE+1)); echo "  ok  $(date '+%F %T')"
      grep -E "Best|best|Test" "$LOG" | tail -3
    else
      FAIL=$((FAIL+1)); echo "  FAIL rc=$RC"; tail -5 "$LOG"
    fi
  done
done

echo ""
echo "===== 2단계 종료 $(date '+%F %T') ====="
echo "  총 ${TOTAL} / 성공 ${DONE} / 실패 ${FAIL}"
