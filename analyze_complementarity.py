#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complementarity analysis (reviewer R1-7).

 per-modality softmax CSV만으로 계산. 모델 forward 없음, GPU 불필요.
    요구: error-overlap matrices, disagreement rates, prediction
correlations, conditional gains when one modality fails.
"""
import argparse, os
import numpy as np
import pandas as pd

MODS = {
    'Nvgestures': [('Color','color'), ('Depth','depth'), ('IR','ir'),
                   ('Normal','normal'), ('Opt.flow','depth_optflow')],
    'Briareo':    [('Color','rgb'), ('Depth','depth'), ('IR','ir'),
                   ('Normal','normal'), ('Opt.flow','rgb_optflow')],
}

def load(csv_dir, pairs):
    P, names = {}, []
    for disp, fn in pairs:
        p = os.path.join(csv_dir, f'{fn}.csv')
        if os.path.exists(p):
            P[disp] = pd.read_csv(p, header=None).values; names.append(disp)
        else:
            print(f"  [warn] 없음: {p}")
    gt = pd.read_csv(os.path.join(csv_dir, 'original.csv'), header=None).iloc[:, 0].values
    return P, names, gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(MODS))
    ap.add_argument('--csv_dir', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    P, names, gt = load(a.csv_dir, MODS[a.dataset])
    N = len(gt)
    pred  = {m: P[m].argmax(1) for m in names}
    wrong = {m: (pred[m] != gt) for m in names}
    acc   = {m: 100*(1-wrong[m].mean()) for m in names}

    print("=" * 78)
    print(f" Complementarity analysis — {a.dataset}  (N={N}, {len(names)} modalities)")
    print(f" csv_dir: {a.csv_dir}")
    print("=" * 78)

    print("\n[0] Unimodal accuracy")
    for m in names:
        print(f"      {m:10s} {acc[m]:6.2f}%   (errors {int(wrong[m].sum()):4d})")

    hdr = "          " + "".join(f"{m:>10s}" for m in names)

    print("\n[1] Error-overlap matrix — 두 모달리티가 동시에 틀린 비율 (%)")
    print("      값이 클수록 오류가 겹친다 = 상보성 낮음")
    print(hdr)
    for i in names:
        print(f"      {i:>8s}" + "".join(f"{100*(wrong[i]&wrong[j]).mean():10.2f}" for j in names))

    print("\n[2] Conditional error  P(column wrong | row wrong) (%)")
    print("      100에 가까울수록 같은 샘플에서 함께 실패 = 독립 정보 없음")
    print(hdr)
    for i in names:
        den = wrong[i].sum()
        cells = [f"{100*(wrong[i]&wrong[j]).sum()/den:10.2f}" if den else f"{'-':>10s}" for j in names]
        print(f"      {i:>8s}" + "".join(cells))

    print("\n[3] Disagreement rate — 예측이 다른 비율 (%)")
    print(hdr)
    for i in names:
        print(f"      {i:>8s}" + "".join(f"{100*(pred[i]!=pred[j]).mean():10.2f}" for j in names))

    print("\n[4] Q-statistic — 오차 상관 (1=완전 종속, 0=독립)")
    print(hdr)
    for i in names:
        cells = []
        for j in names:
            if i == j:
                cells.append(f"{'-':>10s}"); continue
            ci, cj = ~wrong[i], ~wrong[j]
            n11 = (ci & cj).sum(); n00 = (~ci & ~cj).sum()
            n10 = (ci & ~cj).sum(); n01 = (~ci & cj).sum()
            den = n11*n00 + n01*n10
            cells.append(f"{(n11*n00 - n01*n10)/den:10.3f}" if den else f"{'nan':>10s}")
        print(f"      {i:>8s}" + "".join(cells))

    late = lambda keys: 100*((np.mean([P[k] for k in keys],0).argmax(1) == gt).mean())
    full = late(names)
    print(f"\n[5] Conditional gain — late fusion {full:.2f}% ({len(names)}개) 기준")
    print("      각 모달리티 제거 시 변화. 양수면 없는 편이 낫다")
    for m in names:
        v = late([k for k in names if k != m])
        print(f"      -{m:10s} {v:6.2f}%   ({v-full:+6.2f} pp)")

    any_right = np.zeros(N, bool)
    for m in names: any_right |= ~wrong[m]
    print(f"\n[6] Oracle (하나라도 맞히면 정답) : {100*any_right.mean():.2f}%")
    print(f"    모든 모달리티 동시 오답         : {100*(1-any_right.mean()):.2f}%")
    print(f"    late fusion                     : {full:.2f}%")
    print(f"    → 완벽한 선택기 가정 시 여지    : {100*any_right.mean()-full:.2f} pp")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        pd.DataFrame([[100*(wrong[i]&wrong[j]).mean() for j in names] for i in names],
                     index=names, columns=names).to_csv(
            os.path.join(a.out, f'{a.dataset}_error_overlap.csv'),
            float_format='%.2f', encoding='utf-8-sig')
        print(f"\n  저장: {a.out}/{a.dataset}_error_overlap.csv")

if __name__ == '__main__':
    main()
