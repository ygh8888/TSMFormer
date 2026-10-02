#!/usr/bin/env python3
"""Briareo 시드별 런타임 config.

NVGesture와 달리 데이터셋이 공식 validation 분할을 제공하므로
                                      재학습(D안)이 불필요하다. 기존 백본을 그대로 쓰고
save 경로에만 시드를 반영한다.
"""
import json, sys, os

src, seed, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(src))
base = f"{tag}_s{seed}"

d['checkpoints']['save_name'] = base
d['checkpoints']['save_dir']  = f"./experiments/BL5_revision/fusion/{tag}/seed{seed}/checkpoints/"
d['checkpoints']['tb_path']   = f"./experiments/BL5_revision/fusion/{tag}/seed{seed}/train_log"

# 백본 존재 확인 (경로는 원본 유지)
missing = [m['checkpoint'] for m in d.get('cmaf', {}).get('modalities', [])
           if not os.path.exists(m.get('checkpoint', ''))]
if missing:
    sys.stderr.write("ERROR: 백본 없음:\n  " + "\n  ".join(missing) + "\n")
    sys.exit(1)

out = f"hyperparameters/_runtime/{base}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(d, open(out, 'w'), indent=4)
print(out)
