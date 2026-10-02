"""시드별 런타임 config 생성.

- save_name / save_dir / tb_path에 시드를 반영해 run 간 덮어쓰기를 막는다.
- data.full_train=False (901/149 subject-disjoint 분할)
- cmaf.modalities[*].checkpoint를 재학습 백본(BL5_revision)으로 교체.
  D안: fusion 비교는 validation이 오염되지 않은 백본 위에서 수행해야 한다.
"""
import json, sys, os, re

src, seed, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REV_DIR = "experiments/BL5_revision/NVGestures/checkpoints/NVGestures"

d = json.load(open(src))
base = f"{tag}_s{seed}"

d['checkpoints']['save_name'] = base
d['checkpoints']['save_dir']  = f"./experiments/BL5_revision/fusion/{tag}/seed{seed}/checkpoints/"
d['checkpoints']['tb_path']   = f"./experiments/BL5_revision/fusion/{tag}/seed{seed}/train_log"
d.setdefault('data', {})['full_train'] = False

# 백본 경로 치환 + 존재 확인
missing = []
for m in d.get('cmaf', {}).get('modalities', []):
    name = m.get('name')
    new = f"{REV_DIR}/best_rev_nvgestures_{name}.pth"
    m['checkpoint'] = new
    if not os.path.exists(new):
        missing.append(new)
if missing:
    sys.stderr.write("ERROR: 재학습 백본 없음:\n  " + "\n  ".join(missing) + "\n")
    sys.exit(1)

out = f"hyperparameters/_runtime/{base}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(d, open(out, 'w'), indent=4)
print(out)
