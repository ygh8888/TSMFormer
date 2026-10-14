#!/usr/bin/env python
"""How much is there for a learned combiner to learn?  (no training)

For the frozen controlled-protocol backbones, compares the unimodal posteriors on
the sequences the fusion modules are trained on (train split, evaluation
transform, as in fusion training) with those on validation and test:
per-modality accuracy, agreement among modalities, oracle, confidence.
Also reports leave-one-modality-out late fusion on validation and test.

Usage:  python train_agreement.py [--dataset nvg|bri|all] [--out analysis_fix]
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CFG = {'nvg': 'hyperparameters/_runtime/perclass_fix_s1994.json',
       'bri': 'hyperparameters/_runtime/bri_perclass_fix_s1994.json'}


class _NullWriter:
    def __init__(self, *a, **k): pass
    def add_scalar(self, *a, **k): pass
    def close(self): pass


@torch.no_grad()
def unimodal_probs(tr, loaders, device):
    n = len(loaders[0].dataset)
    assert all(len(l.dataset) == n for l in loaders)
    dls = [DataLoader(l.dataset, batch_size=8, shuffle=False, drop_last=False,
                      num_workers=l.num_workers, pin_memory=True) for l in loaders]
    for m in tr.modal_models:
        m.eval()
    P = [[] for _ in dls]; Y = []
    for bl in zip(*dls):
        labs = [b[1].view(-1) for b in bl]
        for l in labs[1:]:
            assert torch.equal(l, labs[0])
        Y.append(labs[0])
        for i, (m, b) in enumerate(zip(tr.modal_models, bl)):
            _, u = m.module.extract_tokens(b[0].to(device))
            P[i].append(F.softmax(u.float(), 1).cpu())
    return np.stack([torch.cat(p).numpy() for p in P]), torch.cat(Y).numpy()


def summarize(P, y, names):
    pred = P.argmax(2); corr = pred == y[None]
    lf = P.mean(0).argmax(1) == y
    agree_all = (pred == pred[0:1]).all(0)
    r = dict(n=int(len(y)),
             unimodal_acc={nm: round(float(c.mean()) * 100, 2) for nm, c in zip(names, corr)},
             late_fusion=round(float(lf.mean()) * 100, 2),
             oracle=round(float(corr.any(0).mean()) * 100, 2),
             all_correct=round(float(corr.all(0).mean()) * 100, 2),
             all_agree=round(float(agree_all.mean()) * 100, 2),
             any_modality_wrong=round(float((~corr).any(0).mean()) * 100, 2),
             lf_wrong_but_some_modality_right=round(float((~lf & corr.any(0)).mean()) * 100, 2),
             mean_max_prob={nm: round(float(p.max(1).mean()), 3) for nm, p in zip(names, P)})
    loo = {}
    for k, nm in enumerate(names):
        idx = [j for j in range(len(names)) if j != k]
        loo[nm] = round(float((P[idx].mean(0).argmax(1) == y).mean()) * 100, 2)
    r['late_fusion_without'] = loo
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=['nvg', 'bri', 'all'])
    ap.add_argument('--out', default='analysis_fix')
    a = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mod = importlib.import_module('train_perclass'); mod.SummaryWriter = _NullWriter
    res = {}
    for ds in (['nvg', 'bri'] if a.dataset == 'all' else [a.dataset]):
        tr = mod.CMAFTrainer(CFG[ds], device)
        names = [m['name'] for m in tr.modalities]
        res[ds] = {}
        for split, loaders in [('train', tr.train_loaders), ('val', tr.val_loaders), ('test', tr.test_loaders)]:
            P, y = unimodal_probs(tr, loaders, device)
            res[ds][split] = summarize(P, y, names)
            np.savez_compressed(Path(a.out) / f'unimodal_{ds}_{split}_full.npz', prob=P, y=y,
                                names=np.array(names))
            print(f"\n[{ds} / {split}]  n={res[ds][split]['n']}")
            for k, v in res[ds][split].items():
                if k != 'n':
                    print(f'  {k:34s} {v}')
        del tr; torch.cuda.empty_cache()
    json.dump(res, open(Path(a.out) / 'train_agreement.json', 'w'), indent=1)
    print(f"\nsaved {a.out}/train_agreement.json")


if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    main()
