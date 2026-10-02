#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2단계 결과 집계 — 변형별 시드 평균/표준편차 (R1-1, R1-2)."""
import re, glob, os
from collections import defaultdict
import numpy as np

TAGS = ['perclass', 'lora_p0', 'mbt_bl2in', 'mbt_gate', 'dense']
LABEL = {
    'perclass':  'Per-class weighting',
    'lora_p0':   'LoRA ensemble (P0)',
    'mbt_bl2in': 'Bottleneck + stronger input',
    'mbt_gate':  'Bottleneck (MBT+gate)',
    'dense':     'Dense cross-attention',
}

# 로그에서 최종 test 정확도를 뽑는다. 스크립트마다 표기가 달라 여러 패턴 시도.
PATTERNS = [
    r'FINAL TEST:\s*acc=([0-9.]+)',          # train_perclass / cmaf 계열 공통
    r'Test\s*(?:acc|accuracy)\s*[:=]\s*([0-9.]+)',
    r'Accuracy\s*[:=]\s*([0-9.]+)',
]
VAL_PATTERN = r'Best val accuracy:\s*([0-9.]+)'

def extract(path):
    txt = open(path, errors='ignore').read()
    vals = []
    for pat in PATTERNS:
        vals += re.findall(pat, txt, re.I)
        if vals: break
    if not vals:
        return None, None
    v = float(vals[-1]);  v = v*100 if v <= 1.0 else v
    mv = re.findall(VAL_PATTERN, txt, re.I)
    bv = (float(mv[-1])*100 if float(mv[-1]) <= 1.0 else float(mv[-1])) if mv else None
    return v, bv

res = defaultdict(dict)
for f in sorted(glob.glob('logs_revision/stage2_*.log')):
    m = re.match(r'.*stage2_(.+)_s(\d+)\.log', f)
    if not m: continue
    tag, seed = m.group(1), int(m.group(2))
    v, bv = extract(f)
    if v is not None:
        res[tag][seed] = (v, bv)
    else:
        print(f"  [진행중] {tag} seed={seed}")

print("=" * 76)
print(" Stage-2 results — fusion variants (NVGesture, 901/149/482 protocol)")
print("=" * 76)
print(f"{'Variant':<30}{'n':>3}{'mean':>9}{'std':>8}   seeds")
print("-" * 76)
for tag in TAGS:
    d = res.get(tag, {})
    if not d:
        print(f"{LABEL[tag]:<30}{0:>3}{'-':>9}{'-':>8}   (없음)"); continue
    vals = np.array([d[s][0] for s in sorted(d)])
    detail = ", ".join(f"{s}:{d[s][0]:.2f}" for s in sorted(d))
    print(f"{LABEL[tag]:<30}{len(vals):>3}{vals.mean():9.2f}{vals.std(ddof=1) if len(vals)>1 else 0:8.2f}   {detail}")
print("-" * 76)
print("  (late fusion 기준선은 cs.py로 별도 산출)")
