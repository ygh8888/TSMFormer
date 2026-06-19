"""
CMAF v4 — Bottleneck Cross-Modal Fusion (MBT-style) + Hybrid Ensemble Gate

설계 핵심 (v3 대비):
  1. 입력: pooled (B,512) → temporal tokens (B,T,512). 모달마다 T 다를 수 있음
     (color/depth/ir/normal=40, optflow=20). bottleneck이 가변 T를 자연 처리.
  2. dense pairwise cross-attn O(M^2) → shared bottleneck tokens Z 경유 O(M).
     모달끼리 직접 attend 안 함 -> 과적합 차단 + 연산 절감 (MBT, NeurIPS 2021).
  3. Hybrid 앙상블 게이트: fusion logits와 freeze된 단일모달 logits(BL2 헤드)를
     학습 가능 게이트로 결합. 게이트가 fusion=0으로 학습되면 BL2 앙상블로 수렴
     -> 89.63%가 사실상 하한. A-1 교훈(앙상블 경로 버리면 진다)을 구조에 내장.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=4, dropout=0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        qn = self.ln_q(q)
        kvn = self.ln_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        return q + self.dropout(out)


class SelfAttnFFN(nn.Module):
    def __init__(self, d_model=512, n_heads=4, dff=1024, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        zn = self.ln1(z)
        a, _ = self.attn(zn, zn, zn, need_weights=False)
        z = z + self.dropout(a)
        z = z + self.dropout(self.ffn(self.ln2(z)))
        return z


class BottleneckFusionLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=4, dff=1024, dropout=0.1):
        super().__init__()
        self.write = CrossAttnBlock(d_model, n_heads, dropout)
        self.refine = SelfAttnFFN(d_model, n_heads, dff, dropout)

    def forward(self, Z, token_list):
        for tok in token_list:
            Z = self.write(Z, tok)
        Z = self.refine(Z)
        return Z


class CMAFv4Bottleneck(nn.Module):
    def __init__(self, n_modalities, n_classes, d_model=512,
                 n_bottleneck=4, n_layers=2, n_heads=4, dff=1024,
                 dropout=0.3, init_fusion_logit=0.0, freeze_gate=False):
        super().__init__()
        self.n_modalities = n_modalities
        self.d_model = d_model

        self.bottleneck = nn.Parameter(torch.zeros(1, n_bottleneck, d_model))
        nn.init.normal_(self.bottleneck, std=0.02)

        self.layers = nn.ModuleList([
            BottleneckFusionLayer(d_model, n_heads, dff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

        self.gate = nn.Parameter(torch.tensor([init_fusion_logit, 0.0]))
        if freeze_gate:
            self.gate.requires_grad = False

    def forward(self, token_list, unimodal_logits):
        B = token_list[0].size(0)
        Z = self.bottleneck.expand(B, -1, -1)

        for layer in self.layers:
            Z = layer(Z, token_list)

        Z = self.ln_out(Z)
        fused = Z.mean(dim=1)
        logits_fusion = self.classifier(self.dropout(fused))

        ens = torch.stack(unimodal_logits, dim=0).mean(dim=0)

        g = F.softmax(self.gate, dim=0)
        logits = g[0] * logits_fusion + g[1] * ens
        return logits

    @torch.no_grad()
    def gate_report(self):
        g = F.softmax(self.gate, dim=0).cpu().tolist()
        return {'fusion': round(g[0], 4), 'ensemble': round(g[1], 4)}


def build_cmaf_v4(n_modalities, n_classes, **kwargs):
    return CMAFv4Bottleneck(n_modalities, n_classes, **kwargs)
