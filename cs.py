"""
Late Fusion Multi-modal Evaluation Script (Paper Table 1 reproduction)
=======================================================================
Usage:
    python cs.py --dataset Briareo --all
    python cs.py --dataset Nvgestures --all
    python cs.py --dataset Briareo --modalities normal ir depth

Output files:
    results/Briareo_fusion_results.csv
    results/Briareo_fusion_results.txt
    results/Nvgestures_fusion_results.csv
    results/Nvgestures_fusion_results.txt
"""

import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd


DATASET_CONFIG = {
    'Briareo': {
        'csv_dir': 'csv/Briareo',
        'n_samples': 288,
        'modalities': ['rgb', 'depth', 'ir', 'normal', 'rgb_optflow'],
        'modality_labels': ['Color', 'Depth', 'IR', 'Normals', 'Optical flow'],
    },
    'Nvgestures': {
        'csv_dir': 'csv/Nvgestures',
        'n_samples': 482,
        'modalities': ['color', 'depth', 'ir', 'normal', 'depth_optflow'],
        'modality_labels': ['Color', 'Depth', 'IR', 'Normals', 'Optical flow'],
    }
}


def load_csv(csv_dir, modality):
    path = os.path.join(csv_dir, f'{modality}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, header=None).values.tolist()


def load_gt(csv_dir):
    path = os.path.join(csv_dir, 'original.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    return pd.read_csv(path, header=None).iloc[:, 0].values.tolist()


def late_fusion(prob_list):
    n = len(prob_list)
    return [
        np.add.reduce([prob_list[m][i] for m in range(n)]) / n
        for i in range(len(prob_list[0]))
    ]


def evaluate(fused, gt):
    c = sum(1 for i in range(len(fused)) if np.argmax(fused[i]) == gt[i])
    return c / len(gt)


def run_all_combinations(dataset, args):
    cfg = DATASET_CONFIG[dataset]
    csv_dir = args.csv_dir if args.csv_dir else cfg['csv_dir']
    modalities = cfg['modalities']
    labels = cfg['modality_labels']

    gt = load_gt(csv_dir)

    available = {}
    print(f"\nChecking available CSV files for {dataset}:")
    for m, l in zip(modalities, labels):
        data = load_csv(csv_dir, m)
        if data is not None:
            available[m] = data
            print(f"  [OK]   {l} ({m}.csv)")
        else:
            print(f"  [SKIP] {l} ({m}.csv not found)")

    if not available:
        print("No CSV files available for evaluation.")
        return

    avail_keys = list(available.keys())
    rows = []

    print(f"\n{'='*72}")
    print(f" Dataset: {dataset}  |  Available modalities: {len(avail_keys)}")
    print(f"{'='*72}")
    print(f" {'#':<3} {'Color':<8} {'Depth':<8} {'IR':<5} {'Normals':<10} {'Opt.Flow':<11} {'Accuracy'}")
    print(f" {'-'*68}")

    for r in range(1, len(avail_keys) + 1):
        for combo in combinations(avail_keys, r):
            prob_list = [available[m] for m in combo]
            fused = late_fusion(prob_list)
            acc = evaluate(fused, gt)

            combo_labels = [labels[modalities.index(m)] for m in combo]
            label_str = ' + '.join(combo_labels)

            row = {
                '#': r,
                'Color':        'v' if 'rgb'   in combo or 'color'        in combo else '',
                'Depth':        'v' if 'depth'  in combo else '',
                'IR':           'v' if 'ir'     in combo else '',
                'Normals':      'v' if 'normal' in combo else '',
                'Optical flow': 'v' if 'rgb_optflow' in combo or 'depth_optflow' in combo else '',
                'Combination':  label_str,
                'Accuracy':     f"{acc*100:.2f}%",
                'Accuracy_float': round(acc * 100, 2),
            }
            rows.append(row)

            print(
                f" {r:<3} "
                f"{'v' if row['Color']        else ' ':<8}"
                f"{'v' if row['Depth']        else ' ':<8}"
                f"{'v' if row['IR']           else ' ':<5}"
                f"{'v' if row['Normals']      else ' ':<10}"
                f"{'v' if row['Optical flow'] else ' ':<11}"
                f"{acc*100:.2f}%"
            )

    print(f" {'='*68}")

    os.makedirs(args.results_dir, exist_ok=True)

    csv_out = f'{args.results_dir}/{dataset}_fusion_results.csv'
    df = pd.DataFrame(rows).drop(columns=['Accuracy_float'])
    df.to_csv(csv_out, index=False, encoding='utf-8-sig')

    txt_out = f'{args.results_dir}/{dataset}_fusion_results.txt'
    with open(txt_out, 'w') as f:
        f.write(f"Table. Results for different modalities on {dataset} dataset.\n")
        f.write(f"Late Fusion (average of softmax probabilities)\n")
        f.write("=" * 72 + "\n")
        f.write(f" {'#':<3} {'Color':<8} {'Depth':<8} {'IR':<5} {'Normals':<10} {'Opt.Flow':<11} {'Accuracy'}\n")
        f.write(" " + "-" * 68 + "\n")
        for row in rows:
            f.write(
                f" {row['#']:<3} "
                f"{'v' if row['Color']        else ' ':<8}"
                f"{'v' if row['Depth']        else ' ':<8}"
                f"{'v' if row['IR']           else ' ':<5}"
                f"{'v' if row['Normals']      else ' ':<10}"
                f"{'v' if row['Optical flow'] else ' ':<11}"
                f"{row['Accuracy']}\n"
            )
        f.write("=" * 72 + "\n")
        best = max(rows, key=lambda x: x['Accuracy_float'])
        f.write(f"\nBest accuracy   : {best['Accuracy']}\n")
        f.write(f"Best combination: {best['Combination']}\n")

    print(f"\nResults saved:")
    print(f"  CSV : {csv_out}")
    print(f"  TXT : {txt_out}")

    best = max(rows, key=lambda x: x['Accuracy_float'])
    print(f"\nBest accuracy : {best['Accuracy']}  ({best['Combination']})")

    return rows


def run_single(dataset, modalities_input, csv_dir=None):
    cfg = DATASET_CONFIG[dataset]
    csv_dir = csv_dir if csv_dir else cfg['csv_dir']
    gt = load_gt(csv_dir)

    prob_list = []
    for m in modalities_input:
        data = load_csv(csv_dir, m)
        if data is None:
            print(f"ERROR: {m}.csv not found")
            return
        prob_list.append(data)

    fused = late_fusion(prob_list)
    acc = evaluate(fused, gt)
    print(f"[{len(modalities_input)} modality] {' + '.join(modalities_input)}: {acc*100:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(
        description='Late Fusion multi-modal evaluation (Paper Table 1 reproduction)')
    parser.add_argument('--csv_dir', type=str, default=None,
                        help='CSV 디렉토리 경로 (미지정시 기본값 사용)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['Briareo', 'Nvgestures'],
                        help='Dataset name')
    parser.add_argument('--modalities', nargs='+', default=None,
                        help='List of modalities (e.g. normal depth ir)')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all combinations and save results')
    args = parser.parse_args()

    if args.all:
        run_all_combinations(args.dataset, args)
    elif args.modalities:
        run_single(args.dataset, args.modalities, args.csv_dir)
    else:
        print("Please specify --modalities or --all.")
        parser.print_help()


if __name__ == '__main__':
    main()
