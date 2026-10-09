#!/usr/bin/env python3
"""Cross-dataset backbone transfer by linear probing (reviewer R2-7).

A unimodal backbone trained on a SOURCE benchmark is frozen; its pooled 512-d feature is
extracted on the TARGET benchmark and only a new linear classifier is trained on the target
training split, selected on the target validation split and evaluated once on the target test
split. The same procedure applied to the target's own backbone gives the in-domain reference.

Usage
  python transfer_probe.py extract --src nvgestures --tgt briareo --mod depth
  python transfer_probe.py probe   --src nvgestures --tgt briareo --mod depth
  python transfer_probe.py summary
Outputs go to experiments/BL6_transfer/.
"""
import argparse, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import imgaug.augmenters as iaa

from utils.configer import Configer
from models.temporal import GestureTransoformer
from datasets.Briareo import Briareo
from datasets.NVGestures import NVGesture

OUT = Path('experiments/BL6_transfer')
SEEDS = [1994, 2024, 777]
MODS = ['color', 'depth', 'ir']                      # measured modalities only
# per benchmark: config used to build the model / data, checkpoint, data_type name
BENCH = {
    'nvgestures': dict(
        cfg='hyperparameters/NVGestures_rev/train_{m}_rev.json',
        ckpt='experiments/BL5_revision/NVGestures/checkpoints/NVGestures/best_rev_nvgestures_{m}.pth',
        dtype={'color': 'color', 'depth': 'depth', 'ir': 'ir'},
        dataset=NVGesture, crop=iaa.CenterCropToFixedSize(256, 192),
        expect={'train': 901, 'val': 149, 'test': 482}),
    'briareo': dict(
        cfg='hyperparameters/Briareo/train_{m}_tsm.json',
        ckpt='experiments/BL2_tsm/Briareo/checkpoints/Briareo/best_tsm_briareo_{m}.pth',
        dtype={'color': 'rgb', 'depth': 'depth', 'ir': 'ir'},
        dataset=Briareo, crop=iaa.CenterCropToFixedSize(200, 200),
        expect={'test': 288}),
}

def configer_for(path):
    ns = argparse.Namespace(hypes=path, phase='train', gpu=[0], resume=None, nogesture=False,
                            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    return Configer(ns)

def bench_files(bench, mod):
    b = BENCH[bench]
    m = b['dtype'][mod]
    return b['cfg'].format(m=m), b['ckpt'].format(m=m), m

def build_backbone(bench, mod, device):
    cfg_path, ckpt_path, _ = bench_files(bench, mod)
    c = configer_for(cfg_path)
    in_planes = 1 if mod in ('depth', 'ir') else 3
    net = GestureTransoformer(c.get('network', 'backbone'), in_planes, c.get('data', 'n_classes'),
                              pretrained=False,
                              n_head=c.get('network', 'n_head'),
                              dropout_backbone=c.get('network', 'dropout2d'),
                              dropout_transformer=c.get('network', 'dropout1d'),
                              dff=c.get('network', 'ff_size'),
                              n_module=c.get('network', 'n_module'),
                              use_tsm=c.get('network', 'use_tsm'),
                              n_frames=c.get('data', 'n_frames'),
                              use_mspe=c.get('network', 'use_mspe') or False,
                              modality_id=c.get('network', 'modality_id') or 0)
    sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    print(f'  backbone  {bench}/{mod}: {ckpt_path}')
    return net.to(device).eval()

@torch.no_grad()
def extract(args):
    torch.manual_seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = build_backbone(args.src, args.mod, device)
    cfg_path, _, dtype = bench_files(args.tgt, args.mod)
    c = configer_for(cfg_path)                       # target data settings (splits, paths, frames)
    b = BENCH[args.tgt]
    out = OUT / 'features'; out.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        f = out / f'{args.src}_on_{args.tgt}_{args.mod}_{split}.pt'
        if f.exists():
            print(f'  skip {f.name}'); continue
        ds = b['dataset'](c, c.get('data', 'data_path'), split=split, data_type=dtype,
                          transforms=b['crop'], n_frames=c.get('data', 'n_frames'), optical_flow=False)
        n = len(ds)
        exp = b['expect'].get(split)
        print(f'  {args.tgt} {split}: {n} sequences' + (f' (expected {exp})' if exp else ''))
        if exp is not None and n != exp:
            sys.exit(f'ABORT: {args.tgt} {split} has {n} sequences, expected {exp} -- check the split files')
        dl = DataLoader(ds, batch_size=8, shuffle=False, drop_last=False, num_workers=4)
        feats, labels = [], []
        for t in dl:
            feats.append(net.extract_feature(t[0].to(device)).float().cpu())
            labels.append(t[1].view(-1).long())
        torch.save({'x': torch.cat(feats), 'y': torch.cat(labels)}, f)
        print(f'  saved {f.name}  {tuple(torch.cat(feats).shape)}')

def train_probe(tr, va, te, n_cls, seed, epochs=300):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mu, sd = tr['x'].mean(0), tr['x'].std(0) + 1e-6
    z = lambda d: (d['x'] - mu) / sd
    xtr, xva, xte = z(tr), z(va), z(te)
    head = nn.Linear(xtr.shape[1], n_cls)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best = (-1.0, None)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); lossf(head(xtr[idx]), tr['y'][idx]).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            va_acc = (head(xva).argmax(1) == va['y']).float().mean().item()
            if va_acc > best[0]:
                best = (va_acc, torch.softmax(head(xte), 1).clone(), ep + 1)
    va_acc, te_prob, ep = best
    te_acc = (te_prob.argmax(1) == te['y']).float().mean().item()
    return va_acc, te_acc, ep, te_prob

def probe(args):
    d = OUT / 'features'
    load = lambda s: torch.load(d / f'{args.src}_on_{args.tgt}_{args.mod}_{s}.pt')
    tr, va, te = load('train'), load('val'), load('test')
    n_cls = int(max(tr['y'].max(), va['y'].max(), te['y'].max())) + 1
    res = {'src': args.src, 'tgt': args.tgt, 'mod': args.mod, 'n_classes': n_cls,
           'n': {'train': len(tr['y']), 'val': len(va['y']), 'test': len(te['y'])}, 'runs': []}
    probs = OUT / 'probs'; probs.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        va_acc, te_acc, ep, p = train_probe(tr, va, te, n_cls, s)
        torch.save({'p': p, 'y': te['y']}, probs / f'{args.src}_on_{args.tgt}_{args.mod}_s{s}.pt')
        res['runs'].append({'seed': s, 'val': va_acc, 'test': te_acc, 'epoch': ep})
        print(f'  seed {s}: val {va_acc*100:.2f}  test {te_acc*100:.2f}  (epoch {ep})')
    t = np.array([r['test'] for r in res['runs']]) * 100
    res['test_mean'], res['test_sd'] = float(t.mean()), float(t.std(ddof=1))
    print(f'  {args.src} -> {args.tgt} [{args.mod}]  test {t.mean():.2f} +- {t.std(ddof=1):.2f}')
    (OUT / 'results').mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'results' / f'{args.src}_on_{args.tgt}_{args.mod}.json', 'w'), indent=1)

def summary(_):
    rows = []
    for f in sorted((OUT / 'results').glob('*.json')):
        r = json.load(open(f)); rows.append(r)
        print(f"{r['src']:>10s} -> {r['tgt']:<10s} {r['mod']:<6s} "
              f"test {r['test_mean']:6.2f} +- {r['test_sd']:4.2f}   n={r['n']}")
    # late fusion of the three measured modalities, per seed
    print('\nlate fusion of probes (color+depth+ir), mean +- sd over seeds')
    for src in BENCH:
        for tgt in BENCH:
            accs = []
            for s in SEEDS:
                fs = [OUT / 'probs' / f'{src}_on_{tgt}_{m}_s{s}.pt' for m in MODS]
                if not all(f.exists() for f in fs): break
                d = [torch.load(f) for f in fs]
                p = sum(x['p'] for x in d) / len(d)
                accs.append((p.argmax(1) == d[0]['y']).float().mean().item() * 100)
            if len(accs) == len(SEEDS):
                a = np.array(accs)
                print(f'{src:>10s} -> {tgt:<10s} fusion  test {a.mean():6.2f} +- {a.std(ddof=1):4.2f}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['extract', 'probe', 'summary'])
    ap.add_argument('--src', choices=list(BENCH)); ap.add_argument('--tgt', choices=list(BENCH))
    ap.add_argument('--mod', choices=MODS)
    a = ap.parse_args()
    {'extract': extract, 'probe': probe, 'summary': summary}[a.cmd](a)
