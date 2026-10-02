import argparse
import os
import numpy as np
import pandas as pd


DATASET_CONFIG = {
    'Briareo': {
        'csv_dir': 'csv/Briareo',
        'unimodal': ['rgb', 'depth', 'ir', 'normal', 'rgb_optflow'],
    },
    'Nvgestures': {
        'csv_dir': 'csv/Nvgestures',
        'unimodal': ['color', 'depth', 'ir', 'normal', 'depth_optflow'],
    },
}


def load_probs(csv_dir, name):
    path = os.path.join(csv_dir, f'{name}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, header=None).values


def load_gt(csv_dir):
    path = os.path.join(csv_dir, 'original.csv')
    return pd.read_csv(path, header=None).iloc[:, 0].values


def accuracy(probs, gt):
    return float((np.argmax(probs, axis=1) == gt).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['Briareo', 'Nvgestures'])
    ap.add_argument('--csv_dir', default=None)
    ap.add_argument('--cmaf_csv', default='cmaf')
    ap.add_argument('--unimodal', nargs='+', default=None)
    ap.add_argument('--step', type=float, default=0.05)
    args = ap.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    csv_dir = args.csv_dir or cfg['csv_dir']
    uni_names = args.unimodal or cfg['unimodal']

    gt = load_gt(csv_dir)
    N = len(gt)

    uni_probs = []
    print(f"\n[{args.dataset}] 단일모달 csv 로드:")
    for name in uni_names:
        p = load_probs(csv_dir, name)
        if p is None:
            print(f"  [SKIP] {name}.csv 없음")
            continue
        assert p.shape[0] == N, f"{name}.csv 행수({p.shape[0]}) != GT({N})"
        uni_probs.append(p)
        print(f"  [OK]   {name}.csv  shape={p.shape}")
    if not uni_probs:
        print("단일모달 csv가 하나도 없습니다.")
        return
    late_mean = np.mean(np.stack(uni_probs, axis=0), axis=0)

    cmaf = load_probs(csv_dir, args.cmaf_csv)
    if cmaf is None:
        print(f"\n[오류] {args.cmaf_csv}.csv 가 없습니다. 먼저 export 하세요.")
        return
    assert cmaf.shape[0] == N, f"cmaf 행수({cmaf.shape[0]}) != GT({N})"
    assert cmaf.shape[1] == late_mean.shape[1], "클래스 수 불일치"

    acc_late = accuracy(late_mean, gt)
    acc_cmaf = accuracy(cmaf, gt)
    print(f"\n{'='*56}")
    print(f"  기준점")
    print(f"  late fusion (alpha=0.0) : {acc_late*100:.2f}%   [{len(uni_probs)} modalities]")
    print(f"  CMAF-only   (alpha=1.0) : {acc_cmaf*100:.2f}%")
    print(f"{'='*56}")

    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, args.step), 4)
    results = {}
    print(f"\n  alpha   accuracy")
    print(f"  {'-'*22}")
    for a in alphas:
        blended = a * cmaf + (1.0 - a) * late_mean
        acc = accuracy(blended, gt)
        results[float(a)] = acc
        print(f"  {a:>4.2f}   {acc*100:6.2f}%")

    best_a = max(results, key=results.get)
    best_acc = results[best_a]
    print(f"  {'-'*22}")
    print(f"\n  >>> best alpha = {best_a:.2f}   acc = {best_acc*100:.2f}%")
    print(f"      late 대비 {(best_acc-acc_late)*100:+.2f}%p, CMAF 대비 {(best_acc-acc_cmaf)*100:+.2f}%p")

    if best_a <= args.step + 1e-9:
        print(f"\n  [해석] best가 alpha~0 -> 현 CMAF는 앙상블에 유의미한 가치를 못 더함.")
        print(f"         (B)bottleneck/(D)adapter 단계 진행 근거.")
    elif best_a >= 1.0 - args.step - 1e-9:
        print(f"\n  [해석] best가 alpha~1 -> CMAF 단독이 최선.")
    else:
        print(f"\n  [해석] 중간 alpha에서 best -> CMAF와 앙상블이 상보적.")
        print(f"         fusion이 앙상블을 포함/확장함을 보이는 결과.")

    os.makedirs('results', exist_ok=True)
    out = os.path.join('results', f'{args.dataset}_blend_sweep.csv')
    pd.DataFrame(
        [{'alpha': a, 'accuracy_pct': round(results[a]*100, 2)} for a in alphas]
    ).to_csv(out, index=False)
    print(f"\n  저장: {out}")


if __name__ == '__main__':
    main()
