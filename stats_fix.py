#!/usr/bin/env python
"""Paired statistics for the *_fix fusion runs against late fusion (CPU only).

Inputs (from eval_fix_preds.py):  <dir>/unimodal_<ds>_{val,test}.npz, <dir>/preds/*.npz
Reports
  1. late fusion: five modalities, and the subset chosen on the full validation set
     (highest validation accuracy, ties broken by lowest validation NLL)
  2. per run: test accuracy, exact McNemar test against five-modality late fusion
  3. per variant: seed-mean difference with a paired bootstrap 95% CI over test
     sequences (10,000 resamples), two-sided bootstrap p, Holm correction
Usage:  python stats_fix.py [--dir analysis_fix] [--B 10000]
"""
import argparse, glob, itertools, json, math, os, re
from collections import defaultdict
import numpy as np


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def lf(prob, idx):
    return prob[list(idx)].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='analysis_fix')
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    report = {}
    for ds in ['nvg', 'bri']:
        fv, ft = f'{a.dir}/unimodal_{ds}_val.npz', f'{a.dir}/unimodal_{ds}_test.npz'
        if not (os.path.exists(fv) and os.path.exists(ft)):
            continue
        V, T = np.load(fv), np.load(ft)
        names = [str(x) for x in T['names']]; M = len(names)
        yv, yt = V['y'], T['y']
        print('\n' + '=' * 78 + f'\n{ds.upper()}  val n={len(yv)}  test n={len(yt)}  modalities={names}')

        # ---- late fusion subsets
        subs = []
        for k in range(1, M + 1):
            for idx in itertools.combinations(range(M), k):
                pv, pt = lf(V['prob'], idx), lf(T['prob'], idx)
                subs.append(dict(idx=idx, name='+'.join(names[i] for i in idx),
                                 val=(pv.argmax(1) == yv).mean(),
                                 nll=-np.log(pv[np.arange(len(yv)), yv].clip(1e-12)).mean(),
                                 test=(pt.argmax(1) == yt).mean(),
                                 ct=(pt.argmax(1) == yt)))
        full = next(s for s in subs if len(s['idx']) == M)
        best_val = max(s['val'] for s in subs)
        tied = [s for s in subs if s['val'] == best_val]
        sel = min(tied, key=lambda s: s['nll'])
        best_test = max(subs, key=lambda s: s['test'])
        print(f"LF five modalities : val {full['val']*100:6.2f}  test {full['test']*100:6.2f} ({full['ct'].sum()}/{len(yt)})")
        print(f"LF val-selected    : {sel['name']}  val {sel['val']*100:6.2f}  test {sel['test']*100:6.2f} ({sel['ct'].sum()})"
              f"   [{len(tied)} subsets tied at best val; their test range "
              f"{min(s['test'] for s in tied)*100:.2f}-{max(s['test'] for s in tied)*100:.2f}]")
        print(f"LF test-selected   : {best_test['name']}  test {best_test['test']*100:6.2f}  (reference only)")
        oracle = (T['prob'].argmax(2) == yt[None]).any(0).mean()
        print(f"Oracle (any modality correct): {oracle*100:.2f}")

        # ---- runs
        runs = defaultdict(list)
        for f in sorted(glob.glob(f'{a.dir}/preds/*.npz')):
            tag, s = re.match(r'(.+)_s(\d+)\.npz', os.path.basename(f)).groups()
            P = np.load(f)
            if len(P['test_y']) != len(yt) or not np.array_equal(P['test_y'], yt):
                continue
            runs[tag].append(dict(seed=int(s), ct=(P['test_prob'].argmax(1) == yt),
                                  cv=(P['val_prob'].argmax(1) == P['val_y']), ep=int(P['epoch'])))
        print(f"\n{'variant':16s}{'seed':>6s}{'ep':>4s}{'val':>8s}{'test':>8s}{'Δ(seq)':>8s}{'b':>4s}{'c':>4s}{'McNemar p':>11s}")
        summ = []
        for tag, rs in runs.items():
            for r in sorted(rs, key=lambda r: r['seed']):
                b = int((r['ct'] & ~full['ct']).sum()); c = int((~r['ct'] & full['ct']).sum())
                r.update(b=b, c=c, p=mcnemar_exact(b, c))
                print(f"{tag:16s}{r['seed']:>6d}{r['ep']:>4d}{r['cv'].mean()*100:>8.2f}{r['ct'].mean()*100:>8.2f}"
                      f"{b-c:>+8d}{b:>4d}{c:>4d}{r['p']:>11.3f}")
            acc = np.array([r['ct'].mean() for r in rs]) * 100
            for ref_name, ref in [('LF5', full['ct']), ('LFval', sel['ct'])]:
                d = np.stack([r['ct'] for r in rs]).mean(0) - ref     # per-sequence, seed-averaged
                boot = d[rng.integers(0, len(d), (a.B, len(d)))].mean(1) * 100
                p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
                summ.append(dict(dataset=ds, variant=tag, ref=ref_name, n_seeds=len(rs),
                                 mean=acc.mean(), sd=acc.std(ddof=1) if len(acc) > 1 else 0.0,
                                 diff=d.mean() * 100, ci_lo=np.percentile(boot, 2.5),
                                 ci_hi=np.percentile(boot, 97.5), p_boot=p,
                                 mcnemar_p=[round(r['p'], 3) for r in rs]))
        for ref_name in ['LF5', 'LFval']:
            grp = [s for s in summ if s['ref'] == ref_name]
            if grp:
                for s, pa in zip(grp, holm(np.array([s['p_boot'] for s in grp]))):
                    s['p_holm'] = float(pa)
                lab = 'five-modality LF' if ref_name == 'LF5' else f"val-selected LF ({sel['name']})"
                print(f"\nvs {lab}")
                print(f"{'variant':16s}{'n':>3s}{'mean±sd':>15s}{'Δpp':>8s}{'95% CI':>18s}{'p':>8s}{'p_Holm':>8s}")
                for s in grp:
                    print(f"{s['variant']:16s}{s['n_seeds']:>3d}{s['mean']:>9.2f}±{s['sd']:<5.2f}{s['diff']:>+8.2f}"
                          f"  [{s['ci_lo']:+6.2f},{s['ci_hi']:+6.2f}]{s['p_boot']:>8.3f}{s['p_holm']:>8.3f}")
        report[ds] = dict(lf5=dict(val=full['val'], test=full['test']),
                          lf_val_selected=dict(name=sel['name'], val=sel['val'], test=sel['test'],
                                               n_tied=len(tied)),
                          lf_test_selected=dict(name=best_test['name'], test=best_test['test']),
                          oracle=oracle, summary=summ)
    with open(f'{a.dir}/stats_summary.json', 'w') as f:
        json.dump(report, f, indent=1, default=float)
    print(f'\nsaved {a.dir}/stats_summary.json')


if __name__ == '__main__':
    main()
