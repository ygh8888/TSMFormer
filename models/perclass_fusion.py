"""per-class modality weighting (LoRA 없음). 학습 파라미터 = W(M×C)뿐."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerClassFusion(nn.Module):
    def __init__(self, n_modalities, n_classes):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(n_modalities, n_classes))
        self.n_modalities = n_modalities
        self.n_classes = n_classes

    def forward(self, unimodal_logits):
        # Combine per-modality *probabilities*, not raw logits, so that W = 0
        # exactly reproduces the simple average of Eq. (1). Weighting logits
        # would not, because softmax is non-linear (reviewer comment R1-6).
        # Returns log-probabilities -> use NLLLoss (argmax is unaffected by log).
        p = F.softmax(torch.stack(unimodal_logits, dim=0), dim=-1)  # (M, B, C)
        w = F.softmax(self.W, dim=0).unsqueeze(1)                   # (M, 1, C)
        return torch.log((w * p).sum(dim=0).clamp_min(1e-12))       # (B, C)

    @torch.no_grad()
    def weight_report(self):
        w = F.softmax(self.W, dim=0)
        return [round(x, 3) for x in w.mean(dim=1).cpu().tolist()]


def build_perclass(n_modalities, n_classes, **kwargs):
    return PerClassFusion(n_modalities, n_classes)
