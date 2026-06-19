"""
CMAF Training Script (Stage 2)

 학습된 단일 모달리티 모델(backbone + GestFormer)을 freeze하고
CMAF 모듈만 학습합니다.

Usage:
    python train_cmaf.py --hypes hyperparameters/Briareo/train_cmaf.json
    python train_cmaf.py --hypes hyperparameters/NVGestures/train_cmaf.json
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.configer import Configer
from models.temporal import GestureTransoformer
from models.perclass_fusion import build_perclass

# 데이터셋 임포트
from datasets.Briareo import Briareo
from datasets.NVGestures import NVGesture
import imgaug.augmenters as iaa

SEED = 1994
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)


def load_single_modal_model(cfg: dict, device) -> nn.Module:
    """단일 모달리티 모델 로드 후 freeze."""
    model = GestureTransoformer(
        cfg['backbone'],
        cfg['in_planes'],
        cfg['n_classes'],
        pretrained=False,
        n_head=cfg['n_head'],
        dropout_backbone=cfg['dropout2d'],
        dropout_transformer=cfg['dropout1d'],
        dff=cfg['ff_size'],
        n_module=cfg['n_module'],
        use_tsm=cfg['use_tsm'],
        n_frames=cfg['n_frames'],
        use_mspe=cfg['use_mspe'],
        modality_id=cfg['modality_id'],
    )
    ckpt = torch.load(cfg['checkpoint'], map_location='cpu',
                      weights_only=False)
    state = {k.replace('module.', ''): v
             for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state)

    # backbone freeze: CMAF 학습 시 backbone 파라미터 고정
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return nn.DataParallel(model).to(device)


class CMAFTrainer:
    def __init__(self, config_path: str, device):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)

        self.device    = device
        self.dataset   = self.cfg['dataset'].lower()
        self.n_classes = self.cfg['data']['n_classes']
        self.modalities = self.cfg['cmaf']['modalities']  # list of modal configs

        # ── 단일 모달리티 모델 로드 (freeze) ──────────────────────────
        print("Loading pre-trained modality models...")
        self.modal_models = []
        for m_cfg in self.modalities:
            model = load_single_modal_model(m_cfg, device)
            self.modal_models.append(model)
            print(f"  Loaded: {m_cfg['name']} ({m_cfg['checkpoint']})")

        # ── CMAF 모듈 초기화 ───────────────────────────────────────────
        self.fusion = build_perclass(
            n_modalities=len(self.modalities),
            n_classes=self.n_classes,
        ).to(device)

        # ── 옵티마이저: per-class W (125개)만 ──
        n_w = sum(p.numel() for p in self.fusion.parameters())
        print(f"  Optimizer params: per-class W = {n_w}")
        self.optimizer = torch.optim.AdamW(
            self.fusion.parameters(),
            lr=self.cfg['solver']['base_lr'],
            weight_decay=self.cfg['solver']['weight_decay'],
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg['solver']['decay_steps'],
            gamma=0.1,
        )
        self.criterion = nn.CrossEntropyLoss()

        # ── TensorBoard ───────────────────────────────────────────────
        save_name = self.cfg['checkpoints']['save_name']
        tb_dir = Path(self.cfg['checkpoints']['tb_path']) / self.cfg['dataset'] / save_name
        self.writer = SummaryWriter(str(tb_dir))

        self.best_accuracy = 0.0
        self.iters         = 0

        # ── 데이터셋 ──────────────────────────────────────────────────
        self._init_dataloaders()

    def _get_in_planes(self, m_cfg):
        if m_cfg.get('optical_flow', False):
            return 2
        elif m_cfg['data_type'] in ['depth', 'ir']:
            return 1
        return 3

    def _init_dataloaders(self):
        data_path = self.cfg['data']['data_path']
        batch_size = self.cfg['data']['batch_size']
        n_frames   = self.cfg['data']['n_frames']
        workers    = self.cfg['solver']['workers']

        if self.dataset == 'briareo':
            Dataset = Briareo
            train_tf = iaa.CenterCropToFixedSize(200, 200)
            val_tf   = iaa.CenterCropToFixedSize(200, 200)
        elif self.dataset == 'nvgestures':
            Dataset = NVGesture
            train_tf = iaa.CenterCropToFixedSize(256, 192)
            val_tf   = iaa.CenterCropToFixedSize(256, 192)

        # CMAF는 기준 모달리티(첫 번째)의 데이터로 레이블을 가져옴
        # 각 모달리티 데이터를 동시에 로드하기 위해 모달리티별 DataLoader 구성
        self.train_loaders = []
        self.val_loaders   = []
        self.test_loaders  = []

        for m_cfg in self.modalities:
            dt   = m_cfg['data_type']
            optf = m_cfg.get('optical_flow', False)

            train_loader = DataLoader(
                Dataset(None, data_path, split='train',
                        data_type=dt, transforms=train_tf,
                        n_frames=n_frames, optical_flow=optf),
                batch_size=batch_size, shuffle=True, drop_last=True,
                num_workers=workers, pin_memory=True,
                worker_init_fn=worker_init_fn)

            val_loader = DataLoader(
                Dataset(None, data_path, split='val',
                        data_type=dt, transforms=val_tf,
                        n_frames=n_frames, optical_flow=optf),
                batch_size=batch_size, shuffle=False, drop_last=True,
                num_workers=workers, pin_memory=True,
                worker_init_fn=worker_init_fn)

            test_loader = DataLoader(
                Dataset(None, data_path, split='test',
                        data_type=dt, transforms=val_tf,
                        n_frames=n_frames, optical_flow=optf),
                batch_size=1, shuffle=False, drop_last=True,
                num_workers=workers, pin_memory=True,
                worker_init_fn=worker_init_fn)

            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)
            self.test_loaders.append(test_loader)

        print(f"Train batches: {len(self.train_loaders[0])}")
        print(f"Val   batches: {len(self.val_loaders[0])}")
        print(f"Test  batches: {len(self.test_loaders[0])}")

    def _extract_features(self, batch_list):
        """각 모달리티 모델에서 tokens + unimodal_logits 추출 (no_grad)."""
        token_list = []
        uni_logits = []
        for i, (model, batch) in enumerate(zip(self.modal_models, batch_list)):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                tok, ulog = model.module.extract_tokens(inputs)
            token_list.append(tok)
            uni_logits.append(ulog)
        return token_list, uni_logits

    def _run_epoch(self, loaders, split='train'):
        is_train = (split == 'train')
        if is_train:
            self.fusion.train()
        else:
            self.fusion.eval()

        correct = 0
        total   = 0
        total_loss = 0.0

        zip_loaders = zip(*loaders)
        n_batches   = len(loaders[0])

        with torch.set_grad_enabled(is_train):
            for batch_list in tqdm(zip_loaders, total=n_batches,
                                   desc=f"{split.capitalize()}"):
                labels = batch_list[0][1].to(self.device)
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)

                token_list, uni_logits = self._extract_features(batch_list)
                logits   = self.fusion(uni_logits)
                loss     = self.criterion(logits, labels)
                if is_train:
                    lam1 = self.cfg['cmaf'].get('lambda_m',  0.3)
                    lam2 = self.cfg['cmaf'].get('lambda_kd', 0.5)
                    T    = self.cfg['cmaf'].get('kd_temp',   4.0)
                    # P1: 모달별 CE (LoRA backbone 개별 판별력 유지)
                    if lam1 > 0:
                        loss_m = sum(self.criterion(u, labels)
                                     for u in uni_logits) / len(uni_logits)
                        loss = loss + lam1 * loss_m
                    # P2: 앙상블->약한모달 KD (color=0, ir=2)
                    if lam2 > 0:
                        weak_idx = self.cfg['cmaf'].get('kd_weak_idx', [0, 2])
                        with torch.no_grad():
                            ens_soft = torch.nn.functional.softmax(
                                torch.stack(uni_logits).mean(0) / T, dim=1)
                        loss_kd = sum(
                            torch.nn.functional.kl_div(
                                torch.nn.functional.log_softmax(
                                    uni_logits[i] / T, dim=1),
                                ens_soft, reduction='batchmean') * (T ** 2)
                            for i in weak_idx) / len(weak_idx)
                        loss = loss + lam2 * loss_kd

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.iters += 1

                preds    = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                total_loss += loss.item()

        acc  = correct / total
        loss = total_loss / n_batches
        return acc, loss

    def train(self):
        epochs    = self.cfg['epochs']
        save_dir  = Path(self.cfg['checkpoints']['save_dir']) / self.cfg['dataset']
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = self.cfg['checkpoints']['save_name']

        print(f"\n{'='*60}")
        print(f"  CMAF Training: {self.cfg['dataset']}")
        print(f"  Modalities: {[m['name'] for m in self.modalities]}")
        print(f"  Epochs: {epochs}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_acc, train_loss = self._run_epoch(self.train_loaders, 'train')
            val_acc,   val_loss   = self._run_epoch(self.val_loaders,   'val')

            self.scheduler.step()

            self.writer.add_scalar('train_accuracy', train_acc, self.iters)
            self.writer.add_scalar('val_accuracy',   val_acc,   self.iters)
            self.writer.add_scalar('train_loss',     train_loss, self.iters)
            self.writer.add_scalar('val_loss',       val_loss,   self.iters)

            print(f"  train={train_acc:.4f}  val={val_acc:.4f}")
            print(f"  weights: {self.fusion.weight_report()}")

            # best model 저장
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                ckpt_path = save_dir / f"best_{save_name}.pth"
                torch.save({
                    'epoch':      epoch + 1,
                    'state_dict': self.fusion.state_dict(),
                    'optimizer':  self.optimizer.state_dict(),
                    'accuracy':   val_acc,
                }, str(ckpt_path))
                print(f"  ✓ Saved best model: val={val_acc:.4f}")

        self.writer.close()
        print(f"\nBest val accuracy: {self.best_accuracy:.4f}")

        # 학습 종료 후 best 체크포인트로 test 1회
        best_ckpt = save_dir / f"best_{save_name}.pth"
        if best_ckpt.exists():
            print(f"\nLoading best checkpoint for final test: {best_ckpt}")
            ck = torch.load(str(best_ckpt), map_location=self.device, weights_only=False)
            self.fusion.load_state_dict(ck['state_dict'])
            test_acc, test_loss = self._run_epoch(self.test_loaders, 'test')
            print(f"  FINAL TEST: acc={test_acc:.4f}  (best val={self.best_accuracy:.4f})")
            print(f"  weights: {self.fusion.weight_report()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', required=True, type=str)
    parser.add_argument('--disable-cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')

    torch.autograd.set_detect_anomaly(True)
    trainer = CMAFTrainer(args.hypes, device)
    trainer.train()
