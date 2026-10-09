#!/usr/bin/env bash
# R2-7: cross-dataset backbone transfer (linear probe). Run from the repository root.
set -e
LOG=experiments/BL6_transfer/logs; mkdir -p $LOG
for MOD in color depth ir; do
  for PAIR in "nvgestures briareo" "briareo nvgestures" "briareo briareo" "nvgestures nvgestures"; do
    set -- $PAIR; SRC=$1; TGT=$2
    echo "=== $SRC -> $TGT [$MOD] $(date '+%F %T')"
    python transfer_probe.py extract --src $SRC --tgt $TGT --mod $MOD 2>&1 | tee -a $LOG/${SRC}_on_${TGT}_${MOD}.log
    python transfer_probe.py probe   --src $SRC --tgt $TGT --mod $MOD 2>&1 | tee -a $LOG/${SRC}_on_${TGT}_${MOD}.log
  done
done
python transfer_probe.py summary | tee $LOG/summary.txt
