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
        z = torch.stack(unimodal_logits, dim=0)        # (M, B, C)
        w = F.softmax(self.W, dim=0).unsqueeze(1)      # (M, 1, C)
        return (w * z).sum(dim=0)                       # (B, C)

    @torch.no_grad()
    def weight_report(self):
        w = F.softmax(self.W, dim=0)
        return [round(x, 3) for x in w.mean(dim=1).cpu().tolist()]


def build_perclass(n_modalities, n_classes, **kwargs):
    return PerClassFusion(n_modalities, n_classes)
