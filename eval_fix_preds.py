#!/usr/bin/env python
"""Re-evaluate the *_fix fusion checkpoints and save per-sequence posteriors.

- validation: all sequences (drop_last=False; the training scripts dropped the
  last incomplete batch, so NVGesture validation used 144 of 149 sequences)
- test: all sequences, batch size 1 as in training
- also saves the unimodal posteriors of the frozen backbones (late fusion reference)

Frozen-backbone variants share one backbone pass per split (cached); LoRA runs
restore their adapted color/IR backbones and run their own pass.

Usage:  python eval_fix_preds.py [--dataset nvg|bri|all] [--out analysis_fix]
"""
import argparse, importlib, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

S5 = [1994, 2024, 777, 42, 3407]
S3 = [1994, 2024, 777]
RUNS = {
    'nvg': [('perclass_fix', 'train_perclass', 'perclass', S5),
            ('lora_p0_fix2', 'train_cmaf_v4_lora', 'v4', S5),
            ('mbt_bl2in_fix', 'train_cmaf_v4', 'v4', S5),
            ('mbt_gate_fix', 'train_cmaf_v4', 'v4', S3),
            ('dense_fix', 'train_cmaf', 'dense', S3)],
    'bri': [('bri_perclass_fix', 'train_perclass', 'perclass', S5)],
}


class _NullWriter:
    def __init__(self, *a, **k): pass
    def add_scalar(self, *a, **k): pass
    def close(self): pass


def full_loader(dl, bs):
    return DataLoader(dl.dataset, batch_size=bs, shuffle=False, drop_last=False,
                      num_workers=dl.num_workers, pin_memory=True)


@torch.no_grad()
def backbone_pass(tr, split, device):
    src = tr.val_loaders if split == 'val' else tr.test_loaders
    n = len(src[0].dataset)
    assert all(len(l.dataset) == n for l in src), 'modality datasets differ in length'
    loaders = [full_loader(l, 1 if split == 'test' else 8) for l in src]
    for m in tr.modal_models:
        m.eval()
    toks = [[] for _ in loaders]; ulog = [[] for _ in loaders]; ys = []
    for bl in zip(*loaders):
        labs = [b[1].view(-1) for b in bl]
        for l in labs[1:]:
            assert torch.equal(l, labs[0]), 'label mismatch across modalities'
        ys.append(labs[0])
        for i, (m, b) in enumerate(zip(tr.modal_models, bl)):
            t, u = m.module.extract_tokens(b[0].to(device))
            toks[i].append(t.cpu()); ulog[i].append(u.cpu())
    out = {'tokens': [torch.cat(t) for t in toks],
           'ulog': [torch.cat(u) for u in ulog],
           'y': torch.cat(ys).long()}
    assert len(out['y']) == n
    return out


@torch.no_grad()
def head_pass(tr, kind, cache, device, bs=8):
    head = tr.fusion if kind == 'perclass' else tr.cmaf
    head.eval()
    N = len(cache['y']); probs = []
    for s in range(0, N, bs):
        tl = [t[s:s + bs].to(device) for t in cache['tokens']]
        ul = [u[s:s + bs].to(device) for u in cache['ulog']]
        if kind == 'perclass':
            lg = head(ul)                       # log-probabilities
        elif kind == 'dense':
            feats = [m.module.pool(t).squeeze(1) for m, t in zip(tr.modal_models, tl)]
            lg = head(feats)
        else:
            lg = head(tl, ul)
        probs.append(F.softmax(lg.float(), dim=1).cpu())
    return torch.cat(probs).numpy()


def restore_lora(tr, ck, device):
    if 'modal_state' in ck:
        n = 0
        for m, msd in zip(tr.modal_models, ck['modal_state']):
            if msd is not None:
                m.load_state_dict(msd); n += 1
            m.eval()
        return f'full state of {n} adapted backbones'
    n = 0
    for m, lsd in zip(tr.modal_models, ck['lora_state']):
        pd = dict(m.named_parameters())
        with torch.no_grad():
            for k, v in lsd.items():
                if k in pd:
                    pd[k].copy_(v.to(device)); n += 1
        m.eval()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=['nvg', 'bri', 'all'])
    ap.add_argument('--out', default='analysis_fix')
    args = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); (out / 'preds').mkdir(parents=True, exist_ok=True)
    dsets = ['nvg', 'bri'] if args.dataset == 'all' else [args.dataset]
    rows = []
    for ds in dsets:
        cache = {}
        for tag, script, kind, seeds in RUNS[ds]:
            mod = importlib.import_module(script)
            mod.SummaryWriter = _NullWriter
            for s in seeds:
                cfg_path = f'hyperparameters/_runtime/{tag}_s{s}.json'
                if not os.path.exists(cfg_path):
                    print(f'[missing cfg] {cfg_path}'); continue
                cfg = json.load(open(cfg_path))
                ck_path = (Path(cfg['checkpoints']['save_dir']) / cfg['dataset']
                           / f"best_{cfg['checkpoints']['save_name']}.pth")
                if not ck_path.exists():
                    print(f'[missing ckpt] {ck_path}'); continue
                print(f'\n##### {tag} seed={s}', flush=True)
                tr = mod.CMAFTrainer(cfg_path, device)
                ck = torch.load(str(ck_path), map_location=device, weights_only=False)
                head = tr.fusion if kind == 'perclass' else tr.cmaf
                head.load_state_dict(ck['state_dict'])
                is_lora = 'lora_state' in ck
                if is_lora:
                    print(f'  LoRA params restored: {restore_lora(tr, ck, device)}')
                key = (cfg['dataset'], cfg['data']['n_frames'],
                       tuple((m['name'], m['checkpoint'], m['data_type'],
                              bool(m.get('optical_flow', False))) for m in tr.modalities))
                res = {}
                for split in ('val', 'test'):
                    if is_lora:
                        c = backbone_pass(tr, split, device)
                    else:
                        if (key, split) not in cache:
                            cache[(key, split)] = backbone_pass(tr, split, device)
                            uni = np.stack([F.softmax(u.float(), 1).numpy()
                                            for u in cache[(key, split)]['ulog']])
                            np.savez_compressed(out / f'unimodal_{ds}_{split}.npz', prob=uni,
                                                y=cache[(key, split)]['y'].numpy(),
                                                names=np.array([m['name'] for m in tr.modalities]))
                        c = cache[(key, split)]
                    p = head_pass(tr, kind, c, device)
                    y = c['y'].numpy()
                    res[split] = (p, y)
                (pv, yv), (pt, yt) = res['val'], res['test']
                np.savez_compressed(out / 'preds' / f'{tag}_s{s}.npz', val_prob=pv, val_y=yv,
                                    test_prob=pt, test_y=yt, epoch=ck.get('epoch', -1),
                                    logged_val=ck.get('accuracy', -1.0))
                cv = (pv.argmax(1) == yv); ct = (pt.argmax(1) == yt)
                nb = (len(yv) // 8) * 8            # what the training loop evaluated
                r = dict(tag=tag, seed=s, epoch=ck.get('epoch', -1),
                         logged_val=round(float(ck.get('accuracy', -1)), 4),
                         val_droplast=round(cv[:nb].mean(), 4),
                         val_full=round(cv.mean(), 4), n_val=len(yv),
                         test=round(ct.mean(), 4), test_correct=int(ct.sum()), n_test=len(yt))
                rows.append(r); print('  ', r, flush=True)
                del tr, ck; torch.cuda.empty_cache()
    with open(out / 'eval_summary.json', 'w') as f:
        json.dump(rows, f, indent=1)
    print('\n' + '=' * 90)
    print(f"{'tag':18s}{'seed':>6s}{'ep':>4s}{'val(log)':>10s}{'val(144)':>10s}{'val(all)':>10s}{'test':>8s}{'n':>6s}")
    for r in rows:
        print(f"{r['tag']:18s}{r['seed']:>6d}{r['epoch']:>4d}{r['logged_val']:>10.4f}"
              f"{r['val_droplast']:>10.4f}{r['val_full']:>10.4f}{r['test']*100:>8.2f}{r['test_correct']:>6d}")


if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    main()
