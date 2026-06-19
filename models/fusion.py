"""
CMAF: Cross-Modal Attention Fusion Module (v3 — Modality-Conditioned Projection)

핵심 변경사항 (v2 대비):
    - 독립 W_q/W_k/W_v/W_o (20쌍 × 4행렬 = 21M) →
      공유 W_q/W_k/W_v/W_o (1세트) + 모달리티별 bias (Embedding)
    - 파라미터: 21M → ~1.1M (95% 감소)
    - 모달리티 특성: bias를 통해 보존

수식:
    Q_m = W_q_shared(F_m) + bias_q[m]   ← 공유 projection + 모달리티 조정
    K_n = W_k_shared(F_n) + bias_k[n]
    V_n = W_v_shared(F_n) + bias_v[n]
    A(m→n) = softmax(Q_m · K_nᵀ / √D) · V_n
    w_m = softmax(MLP(F_m))
    F_fused = Σ_m w_m · (F_m + Σ_{n≠m} A(m→n))

Reference:
    - Vaswani et al., "Attention Is All You Need," NeurIPS 2017
    - Liu et al., "TACFN," arXiv 2025
    - CMAF-Net, PMC 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCPCrossModalAttention(nn.Module):
    """
    Modality-Conditioned Projection Cross-Modal Attention.

    공유 W_q/W_k/W_v/W_o + 모달리티별 bias로 특성 보존.

    Args:
        d_model      (int): feature dimension (default: 512)
        n_modalities (int): 전체 모달리티 수 (bias embedding 크기)
        n_heads      (int): attention heads (default: 4)
        dropout      (float): dropout rate
    """

    def __init__(self, d_model: int = 512, n_modalities: int = 5,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_head      = d_model // n_heads
        self.n_modalities = n_modalities

        # ── 공유 projection (모든 모달리티 쌍이 공유) ─────────────────
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # ── 모달리티별 bias (각 모달리티의 방향 조정) ─────────────────
        # bias_q[m]: 모달리티 m이 질문할 때의 방향 조정
        # bias_k[n]: 모달리티 n이 key를 제공할 때의 방향 조정
        # bias_v[n]: 모달리티 n이 value를 제공할 때의 방향 조정
        self.bias_q = nn.Embedding(n_modalities, d_model)
        self.bias_k = nn.Embedding(n_modalities, d_model)
        self.bias_v = nn.Embedding(n_modalities, d_model)

        self.dropout     = nn.Dropout(dropout)
        self.layer_norm  = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)
        # bias는 작은 값으로 초기화 (공유 projection이 주도하도록)
        for emb in [self.bias_q, self.bias_k, self.bias_v]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, F_m: torch.Tensor, F_n: torch.Tensor,
                mod_id_m: int, mod_id_n: int) -> torch.Tensor:
        """
        Args:
            F_m     : (B, D) — query modality feature
            F_n     : (B, D) — key/value modality feature
            mod_id_m: int    — query modality ID
            mod_id_n: int    — key/value modality ID
        Returns:
            (B, D) — F_m enriched by F_n
        """
        B      = F_m.size(0)
        dev    = F_m.device

        id_m = torch.tensor([mod_id_m], device=dev)
        id_n = torch.tensor([mod_id_n], device=dev)

        # 공유 projection + 모달리티별 bias
        Q = self.W_q(F_m) + self.bias_q(id_m)   # (B, D)
        K = self.W_k(F_n) + self.bias_k(id_n)   # (B, D)
        V = self.W_v(F_n) + self.bias_v(id_n)   # (B, D)

        # multi-head reshape
        Q = Q.unsqueeze(1).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.unsqueeze(1).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        V = V.unsqueeze(1).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        scale = self.d_head ** -0.5
        attn  = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn  = F.softmax(attn, dim=-1)
        attn  = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, self.d_model)
        out = self.W_o(out)

        return self.layer_norm(F_m + self.dropout(out))


class ModalityWeighting(nn.Module):
    """
    모달리티 중요도 가중치: w_m = softmax(MLP(F_m))
    경량 MLP: 512 → 64 → 1

    Args:
        d_model      (int): feature dimension
        n_modalities (int): number of modalities
        dropout      (float): dropout rate
    """

    def __init__(self, d_model: int = 512, n_modalities: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of M tensors, each (B, D)
        Returns:
            weights: (B, M) softmax weights
        """
        scores = torch.cat([self.mlp(f) for f in features], dim=1)
        return F.softmax(scores, dim=1)


class CMAFModule(nn.Module):
    """
    Cross-Modal Attention Fusion Module (v3 — MCP).

    단일 아키텍처로 Briareo / NVGestures / EgoGesture 모두 지원.
    데이터셋별 차이는 dropout, lr 등 하이퍼파라미터로만 조정.

    Args:
        n_modalities (int): 모달리티 수
        d_model      (int): feature dimension (default: 512)
        n_classes    (int): 분류 클래스 수
        n_heads      (int): attention heads (default: 4)
        dropout      (float): dropout (Briareo=0.1, NVGestures=0.3)
        modality_ids (list): 각 모달리티의 ID 리스트 (MSPE와 동일 체계)
    """

    def __init__(
        self,
        n_modalities:  int,
        d_model:       int   = 512,
        n_classes:     int   = 25,
        n_heads:       int   = 4,
        dropout:       float = 0.1,
        modality_ids:  list  = None,
        total_modalities: int = 5,  # bias embedding 전체 크기
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self.d_model      = d_model
        self.modality_ids = modality_ids or list(range(n_modalities))

        # MCP cross-attention (단일 모듈, 모든 쌍이 공유)
        self.cross_attn = MCPCrossModalAttention(
            d_model=d_model,
            n_modalities=total_modalities,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.modality_weighting = ModalityWeighting(
            d_model, n_modalities, dropout)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # 단순 Linear 분류기 (과적합 방지)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of M tensors, each (B, D)
                      순서는 modality_ids와 일치
        Returns:
            logits: (B, n_classes)
        """
        M = len(features)
        assert M == self.n_modalities

        # Step 1: MCP cross-modal attention
        enriched = []
        for m in range(M):
            cross_sum = torch.zeros_like(features[m])
            for n in range(M):
                if m != n:
                    attended = self.cross_attn(
                        features[m], features[n],
                        self.modality_ids[m],
                        self.modality_ids[n],
                    )
                    cross_sum = cross_sum + attended
            enriched_m = self.layer_norm(features[m] + cross_sum)
            enriched.append(enriched_m)

        # Step 2: 모달리티 중요도 가중치
        weights = self.modality_weighting(enriched)  # (B, M)

        # Step 3: 가중합
        fused = torch.zeros_like(enriched[0])
        for m in range(M):
            fused = fused + weights[:, m:m+1] * enriched[m]

        # Step 4: dropout + 분류
        return self.classifier(self.dropout(fused))


def build_cmaf(n_modalities: int, n_classes: int,
               d_model: int = 512, n_heads: int = 4,
               dropout: float = 0.1,
               modality_ids: list = None,
               total_modalities: int = 5) -> CMAFModule:
    """
    CMAF 모듈 빌더.

    Usage:
        # Briareo (5 modalities, ids=[0,1,2,3,4])
        cmaf = build_cmaf(5, 12, dropout=0.1,
                          modality_ids=[0,1,2,3,4])

        # NVGestures (5 modalities, ids=[0,1,2,3,4])
        cmaf = build_cmaf(5, 25, dropout=0.3,
                          modality_ids=[0,1,2,3,4])
    """
    return CMAFModule(
        n_modalities=n_modalities,
        d_model=d_model,
        n_classes=n_classes,
        n_heads=n_heads,
        dropout=dropout,
        modality_ids=modality_ids,
        total_modalities=total_modalities,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("CMAF v3 (Modality-Conditioned Projection) Sanity Check")
    print("=" * 60)

    B, D = 8, 512

    # Briareo (5 modalities, 12 classes)
    ids_b = [0, 1, 2, 3, 4]
    cmaf_b = build_cmaf(5, 12, dropout=0.1, modality_ids=ids_b)
    feats_b = [torch.randn(B, D) for _ in range(5)]
    out_b = cmaf_b(feats_b)
    n_b = sum(p.numel() for p in cmaf_b.parameters())

    # NVGestures (5 modalities, 25 classes)
    ids_nv = [0, 1, 2, 3, 4]
    cmaf_nv = build_cmaf(5, 25, dropout=0.3, modality_ids=ids_nv)
    feats_nv = [torch.randn(B, D) for _ in range(5)]
    out_nv = cmaf_nv(feats_nv)
    n_nv = sum(p.numel() for p in cmaf_nv.parameters())

    print(f"Briareo    출력: {list(out_b.shape)}  params={n_b/1e6:.3f}M")
    print(f"NVGestures 출력: {list(out_nv.shape)}  params={n_nv/1e6:.3f}M")
    print(f"\nv2 대비 파라미터: 21.039M → {n_nv/1e6:.3f}M")
    print(f"감소율: {(1 - n_nv/21039000)*100:.1f}%")
    print("=" * 60)
