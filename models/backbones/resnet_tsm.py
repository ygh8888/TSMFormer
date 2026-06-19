"""
TSM-ResNet-18: Temporal Shift Module inserted into ResNet-18 backbone.

Reference:
    Lin et al., "TSM: Temporal Shift Module for Efficient Video Understanding"
    ICCV 2019 / IEEE TPAMI 2022
    https://arxiv.org/abs/1811.08383

Design choices for this project:
    - shift_div = 8  → 1/8 of channels shifted (C//8 forward + C//8 backward)
    - bidirectional  → uses both past (-1) and future (+1) frame information
    - residual TSM   → shift is applied inside the residual branch (before first conv)
                       identity path is kept clean, preserving spatial features
    - inserted in    → ALL BasicBlocks of ResNet-18 (layers 1-4)
    - zero overhead  → +0 parameters, +0 FLOPs (pure tensor indexing)
    - n_frames       → must match training config (40 for GestFormer datasets)

Interface is identical to models/backbones/resnet.py so temporal.py only needs
a one-line swap: backbone = resnet18_tsm instead of resnet18.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


# ──────────────────────────────────────────────────────────────────────────────
# Core shift operation
# ──────────────────────────────────────────────────────────────────────────────

class TemporalShift(nn.Module):
    """
    Residual Temporal Shift Module.

    Applied inside the residual branch (before the first conv layer) so that
    the identity shortcut retains the unmodified spatial features.

    Tensor flow:
        x : (B*T, C, H, W)          ← input (frames packed into batch dim)
          → reshape (B, T, C, H, W)
          → shift channels along T
          → reshape (B*T, C, H, W)  ← ready for Conv2d

    Shift strategy (bidirectional, shift_div=8):
        - ch [0 : C//8]       ← pulled from frame T-1  (past information)
        - ch [C//8 : C//4]    ← pulled from frame T+1  (future information)
        - ch [C//4 : C]       ← unchanged               (current frame)

    Args:
        n_frames  (int): temporal sequence length; must match actual clip length.
        shift_div (int): denominator for shift fraction.
                         shift_div=8 → 1/8 ch forward + 1/8 ch backward.
    """

    def __init__(self, n_frames: int = 40, shift_div: int = 8):
        super().__init__()
        self.n_frames = n_frames
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, c, h, w = x.shape          # n_batch = B * T
        t = self.n_frames
        b = n_batch // t                    # recover batch size B

        # tracing(FlopCountAnalysis 등) 시 n_batch < t 인 경우 shift 생략
        if b == 0:
            return x

        # ── reshape to expose temporal dimension ──────────────────────────────
        x = x.view(b, t, c, h, w)

        # ── prepare output buffer (clone to avoid in-place issues) ───────────
        out = x.clone()

        fold = c // self.shift_div          # number of channels per direction

        # forward shift: ch[0:fold] of frame T ← ch[0:fold] of frame T-1
        # (past frame supplies these channels to the present)
        out[:, 1:,  :fold]       = x[:, :-1, :fold]        # T=1..T-1 ← T-1
        out[:, 0,   :fold]       = 0.0                      # T=0 has no past

        # backward shift: ch[fold:2*fold] of frame T ← ch[fold:2*fold] of T+1
        # (future frame supplies these channels to the present)
        out[:, :-1, fold:2*fold] = x[:, 1:,  fold:2*fold]  # T=0..T-2 ← T+1
        out[:, -1,  fold:2*fold] = 0.0                      # T=T-1 has no future

        # remaining channels ch[2*fold:] are untouched (already copied by clone)

        # ── reshape back to original format ───────────────────────────────────
        out = out.view(n_batch, c, h, w)
        return out

    def __repr__(self):
        return (f"TemporalShift(n_frames={self.n_frames}, "
                f"shift_div={self.shift_div}, "
                f"fold={self.shift_div} → 2/{self.shift_div} channels shifted)")


# ──────────────────────────────────────────────────────────────────────────────
# TSM-enabled BasicBlock
# ──────────────────────────────────────────────────────────────────────────────

class TSMBasicBlock(nn.Module):
    """
    ResNet-18 BasicBlock with Residual Temporal Shift Module.

    Architecture:
        x ──────────────────────────────────────────── identity ──┐
        │                                                          │
        └─► TemporalShift ─► Conv3×3 ─► BN ─► ReLU               ├─► + ─► ReLU
                          ─► Conv3×3 ─► BN ─────────────────────────►
    The shift acts only inside the residual branch, so the identity shortcut
    always sees the unmodified spatial activations.

    Matches the interface of torchvision BasicBlock exactly.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
        n_frames: int = 40,
        shift_div: int = 8,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # ── temporal shift (inserted before first conv) ───────────────────────
        self.temporal_shift = TemporalShift(n_frames=n_frames, shift_div=shift_div)

        # ── spatial convolutions (identical to original BasicBlock) ───────────
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride,
            padding=dilation, groups=groups, bias=False, dilation=dilation
        )
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, groups=groups, bias=False
        )
        self.bn2       = norm_layer(planes)
        self.downsample = downsample
        self.stride    = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # ── residual branch ───────────────────────────────────────────────────
        out = self.temporal_shift(x)    # ← TSM here, before any convolution
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # ── downsample shortcut (if stride > 1 or channel mismatch) ──────────
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Helper: patch a torchvision ResNet by replacing BasicBlocks with TSMBasicBlocks
# ──────────────────────────────────────────────────────────────────────────────

def _make_tsm_layer(layer: nn.Sequential, n_frames: int, shift_div: int) -> nn.Sequential:
    """
    Replace every BasicBlock in a ResNet layer (nn.Sequential) with
    TSMBasicBlock, transferring all weights.

    Only BasicBlock (ResNet-18/34) is supported. Bottleneck (ResNet-50+) is not
    needed for this project.
    """
    new_blocks = []
    for blk in layer:
        if not isinstance(blk, BasicBlock):
            new_blocks.append(blk)
            continue

        tsm_blk = TSMBasicBlock(
            inplanes   = blk.conv1.in_channels,
            planes     = blk.conv1.out_channels,
            stride     = blk.stride,
            downsample = blk.downsample,
            n_frames   = n_frames,
            shift_div  = shift_div,
        )
        # transfer all matching weights from the pretrained BasicBlock
        tsm_blk.conv1.load_state_dict(blk.conv1.state_dict())
        tsm_blk.bn1.load_state_dict(blk.bn1.state_dict())
        tsm_blk.conv2.load_state_dict(blk.conv2.state_dict())
        tsm_blk.bn2.load_state_dict(blk.bn2.state_dict())

        new_blocks.append(tsm_blk)

    return nn.Sequential(*new_blocks)


# ──────────────────────────────────────────────────────────────────────────────
# Public backbone builder — same interface as models/backbones/resnet.py
# ──────────────────────────────────────────────────────────────────────────────

class TSMResNet18(nn.Module):
    """
    ResNet-18 backbone with Temporal Shift Module in every BasicBlock.

    Wraps a torchvision ResNet-18 and replaces all BasicBlocks with
    TSMBasicBlocks. ImageNet pretrained weights are loaded first, then
    TSMBasicBlocks are initialised with those same weights (zero-cost upgrade).

    Input:
        x : (B * n_frames, in_planes, H, W)
            Frames are stacked along the batch dimension, matching the
            GestFormer dataloader convention.

    Output:
        feats : (B * n_frames, feature_dim)
            Per-frame feature vectors, ready for GestFormer transformer blocks.

    Args:
        pretrained (bool)   : load ImageNet weights before inserting TSM.
        in_planes  (int)    : number of input channels (1 for IR/depth, 3 for RGB).
        dropout    (float)  : dropout applied after global average pooling.
        n_frames   (int)    : sequence length; must match dataloader.
        shift_div  (int)    : 1/shift_div channels shifted per direction.
    """

    def __init__(
        self,
        pretrained: bool = True,
        in_planes: int = 3,
        dropout: float = 0.5,
        n_frames: int = 40,
        shift_div: int = 8,
    ):
        super().__init__()
        self.n_frames  = n_frames
        self.shift_div = shift_div

        # ── build torchvision ResNet-18 (with or without ImageNet weights) ────
        weights = "IMAGENET1K_V1" if pretrained else None
        base    = resnet18(weights=weights)

        # ── adapt first conv for non-RGB inputs ──────────────────────────────
        if in_planes != 3:
            # re-create conv1 for the right number of input channels
            old_conv = base.conv1
            base.conv1 = nn.Conv2d(
                in_planes, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            nn.init.kaiming_normal_(base.conv1.weight, mode="fan_out", nonlinearity="relu")

        # ── replace every BasicBlock with TSMBasicBlock ───────────────────────
        base.layer1 = _make_tsm_layer(base.layer1, n_frames, shift_div)
        base.layer2 = _make_tsm_layer(base.layer2, n_frames, shift_div)
        base.layer3 = _make_tsm_layer(base.layer3, n_frames, shift_div)
        base.layer4 = _make_tsm_layer(base.layer4, n_frames, shift_div)

        # ── expose all layers as attributes ──────────────────────────────────
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool

        # ── dropout + final feature projection ───────────────────────────────
        feature_dim = 512   # ResNet-18 final channel count
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(feature_dim, feature_dim)

    def update_n_frames(self, n_frames: int):
        """실제 temporal 길이를 모든 TemporalShift에 동적으로 반영."""
        for m in self.modules():
            if isinstance(m, TemporalShift):
                m.n_frames = n_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B * T, C, H, W)
        Returns:
            feats : (B * T, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # (B*T, 512)
        x = self.dropout(x)
        x = self.fc(x)            # (B*T, 512)
        return x


def resnet18_tsm(
    pretrained: bool = True,
    in_planes: int = 3,
    dropout: float = 0.5,
    n_frames: int = 40,
    shift_div: int = 8,
) -> TSMResNet18:
    """
    Factory function — mirrors the interface of resnet18() in resnet.py.

    Usage in temporal.py:
        from models.backbones.resnet_tsm import resnet18_tsm
        self.backbone = resnet18_tsm(pretrained, in_planes,
                                      dropout=dropout_backbone,
                                      n_frames=n_frames)
    """
    return TSMResNet18(
        pretrained=pretrained,
        in_planes=in_planes,
        dropout=dropout,
        n_frames=n_frames,
        shift_div=shift_div,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("TSM-ResNet-18 sanity check")
    print("=" * 60)

    B, T, C, H, W = 2, 40, 3, 224, 224
    x = torch.randn(B * T, C, H, W)

    # ── build model ──────────────────────────────────────────────────────────
    model = resnet18_tsm(pretrained=False, in_planes=3, dropout=0.5, n_frames=40)
    model.eval()

    # ── parameter count ──────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params / 1e6:.2f} M")

    # ── forward pass ─────────────────────────────────────────────────────────
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    elapsed = time.time() - t0

    print(f"Input  : {list(x.shape)}")
    print(f"Output : {list(out.shape)}   (expected [{B*T}, 512])")
    print(f"Forward: {elapsed*1000:.1f} ms")

    # ── verify TSM was inserted ───────────────────────────────────────────────
    tsm_blocks = [
        m for m in model.modules() if isinstance(m, TemporalShift)
    ]
    print(f"TSMBasicBlocks inserted: {len(tsm_blocks)}")
    print(f"Each block: {tsm_blocks[0]}")

    # ── compare param count with vanilla ResNet-18 ───────────────────────────
    vanilla = resnet18(weights=None)
    n_vanilla = sum(p.numel() for p in vanilla.parameters() if p.requires_grad)
    print(f"\nVanilla ResNet-18 params : {n_vanilla / 1e6:.2f} M")
    print(f"TSM-ResNet-18 params     : {n_params / 1e6:.2f} M")
    print(f"Difference               : {(n_params - n_vanilla) / 1e6:+.4f} M  (expected ~0)")
    print("=" * 60)
