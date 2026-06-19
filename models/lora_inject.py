"""
lora_inject.py — Conv2d/Linear에 LoRA 주입 (본체 freeze, 저랭크 A/B만 학습).
layer4 + fc에만 부착 → 고수준 feature 적응, 과적합·VRAM 억제.
"""
import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.lora_A = nn.Conv2d(base.in_channels, r, kernel_size=base.kernel_size,
                                stride=base.stride, padding=base.padding, bias=False)
        self.lora_B = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


def _replace_conv(module, r, alpha):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, r, alpha))
        else:
            _replace_conv(child, r, alpha)


def inject_lora(backbone, r=4, alpha=8, target_layer='layer4', inject_fc=True):
    for p in backbone.parameters():
        p.requires_grad = False
    layer = getattr(backbone, target_layer, None)
    if layer is not None:
        _replace_conv(layer, r, alpha)
    if inject_fc and hasattr(backbone, 'fc') and isinstance(backbone.fc, nn.Linear):
        backbone.fc = LoRALinear(backbone.fc, r, alpha)
    return sum(p.numel() for p in backbone.parameters() if p.requires_grad)
