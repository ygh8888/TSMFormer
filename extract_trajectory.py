#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""per-class 학습 궤적 추출 — 다중 시드 평균 + 불확실성 밴드 (R1-5).

로그에서 epoch별 (train loss, val acc, modality weights)를 뽑아
시드 평균과 표준편차를 계산한다. 추가 학습 불필요.
"""
import re, glob, os, sys
import numpy as np
import pandas as pd

TAG = sys.argv[1] if len(sys.argv) > 1 else 'perclass'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'experiments/BL5_revision/analysis'

PAT_TV = re.compile(r'train=([0-9.]+)\s+val=([0-9.]+)')
PAT_W  = re.compile(r'weights:\s*\[([0-9.,\s]+)\]')

runs = {}
for f in sorted(glob.glob(f'logs_revision/stage2_{TAG}_s*.log')):
    seed = int(re.search(r'_s(\d+)\.log', f).group(1))
    txt = open(f, errors='ignore').read()
    tv = PAT_TV.findall(txt)
    ws = PAT_W.findall(txt)
    n = min(len(tv), len(ws))
    if n == 0:
        print(f"  [skip] {f}"); continue
    runs[seed] = {
        'train': np.array([float(a) for a, _ in tv[:n]]),
        'val':   np.array([float(b) for _, b in tv[:n]]) * 100,
        'w':     np.array([[float(x) for x in w.split(',')] for w in ws[:n]]),
    }
    print(f"  {f}  → {n} epochs")

if not runs:
    sys.exit("추출된 run 없음")

seeds = sorted(runs)
E = min(len(runs[s]['val']) for s in seeds)
M = runs[seeds[0]]['w'].shape[1]
val = np.stack([runs[s]['val'][:E] for s in seeds])        # (S, E)
W   = np.stack([runs[s]['w'][:E] for s in seeds])          # (S, E, M)

print(f"\n{'='*70}")
print(f" Trajectory — {TAG}  ({len(seeds)} seeds x {E} epochs, {M} modalities)")
print(f"{'='*70}")
print(f"{'ep':>3}{'val mean':>10}{'std':>7}   modality weights (mean +- std)")
for e in range(E):
    wm, wsd = W[:, e].mean(0), W[:, e].std(0, ddof=1) if len(seeds) > 1 else np.zeros(M)
    ws = " ".join(f"{m:.3f}+-{s:.3f}" for m, s in zip(wm, wsd))
    print(f"{e:>3}{val[:,e].mean():10.2f}{val[:,e].std(ddof=1) if len(seeds)>1 else 0:7.2f}   {ws}")

best_e = val.mean(0).argmax()
print(f"\n  평균 val 최고 epoch : {best_e}  ({val[:,best_e].mean():.2f}%)")
print(f"  최종 epoch          : {E-1}  ({val[:,E-1].mean():.2f}%)")
print(f"  최고→최종 변화      : {val[:,E-1].mean()-val[:,best_e].mean():+.2f} pp")
unif = 1.0 / M
dev = np.abs(W - unif).sum(axis=2)                          # (S, E) L1 편차
print(f"  균일분포 L1 편차    : epoch0 {dev[:,0].mean():.4f} → epoch{E-1} {dev[:,E-1].mean():.4f}")

os.makedirs(OUT, exist_ok=True)
rows = []
for e in range(E):
    r = {'epoch': e,
         'val_mean': val[:,e].mean(),
         'val_std': val[:,e].std(ddof=1) if len(seeds)>1 else 0.0}
    for m in range(M):
        r[f'w{m}_mean'] = W[:,e,m].mean()
        r[f'w{m}_std']  = W[:,e,m].std(ddof=1) if len(seeds)>1 else 0.0
    rows.append(r)
p = os.path.join(OUT, f'{TAG}_trajectory.csv')
pd.DataFrame(rows).to_csv(p, index=False, float_format='%.4f', encoding='utf-8-sig')
print(f"\n  저장: {p}")
