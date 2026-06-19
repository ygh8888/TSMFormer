import torch
import torch.nn as nn
from models.backbones.resnet import resnet18
from models.backbones.vgg import vgg16, vgg16_bn
from models.backbones import c3d
from models.backbones.r3d import r3d_18, r2plus1d_18
from models.attention import EncoderSelfAttention

backbone_dict = {'resnet': resnet18,
                 'vgg': vgg16, 'vgg_bn': vgg16_bn,
                 'c3d': c3d,
                 'r3d': r3d_18, 'r2plus1d': r2plus1d_18}

class _GestureTransformer(nn.Module):
    """Multi Modal model for gesture recognition on 3 channel"""
    def __init__(self, backbone: nn.Module, in_planes: int, out_planes: int,
                 pretrained: bool = False, dropout_backbone=0.1, use_tsm=False, n_frames=40, use_mspe=False, modality_id=0,
                 **kwargs):
        super(_GestureTransformer, self).__init__()

        self.in_planes = in_planes
        ## self.backbone = backbone(pretrained, in_planes, dropout=dropout_backbone)
        if use_tsm:
            from models.backbones.resnet_tsm import resnet18_tsm
            self.backbone = resnet18_tsm(
                pretrained, in_planes,
                dropout=dropout_backbone,
                n_frames=n_frames,
            )
        else:
            self.backbone = backbone(pretrained, in_planes, dropout=dropout_backbone)

        self.self_attention = EncoderSelfAttention(512, 64, 64, **kwargs)

        self.pool = nn.AdaptiveAvgPool2d((1, 512))
        # MSPE: modality-specific positional encoding
        self.use_mspe = use_mspe
        self.modality_id = modality_id
        if use_mspe:
            # 5 modalities: 0=rgb/color, 1=depth, 2=ir, 3=normal, 4=optflow
            self.modality_embedding = nn.Embedding(5, 512)
            nn.init.normal_(self.modality_embedding.weight, mean=0, std=0.02)
        self.classifier = nn.Linear(512, out_planes)


    def forward(self, x):
        shape = x.shape
        # print(x.shape)  #8,40,192,256
 
        x = x.view(-1, self.in_planes, x.shape[-2], x.shape[-1])
        # print(x.shape)   #320,1,192,256   b*f, c,h,w

        # 동적 n_frames: optical flow(in_planes=2)는 실제 T=n_frames//in_planes
        n_frames_actual = shape[1] // self.in_planes
        if hasattr(self.backbone, 'update_n_frames'):
            self.backbone.update_n_frames(n_frames_actual)
        x = self.backbone(x)
        # print(x.shape)
        x = x.view(shape[0], shape[1] // self.in_planes, -1)

        # MSPE: backbone feature에 모달리티 임베딩 추가
        if self.use_mspe:
            mod_id = torch.tensor([self.modality_id], device=x.device)
            mod_emb = self.modality_embedding(mod_id)  # (1, 512)
            x = x + mod_emb.unsqueeze(1)              # (B, T, 512) + (1, 1, 512)
        x = self.self_attention(x)

        x = self.pool(x).squeeze(dim=1)
        x = self.classifier(x)
        return x

    def extract_feature(self, x):
        """backbone + MSPE + self_attention + pool 까지만 수행, (B, 512) 반환"""
        shape = x.shape
        x = x.view(-1, self.in_planes, x.shape[-2], x.shape[-1])
        n_frames_actual = shape[1] // self.in_planes
        if hasattr(self.backbone, 'update_n_frames'):
            self.backbone.update_n_frames(n_frames_actual)
        x = self.backbone(x)
        x = x.view(shape[0], shape[1] // self.in_planes, -1)
        if self.use_mspe:
            mod_id = torch.tensor([self.modality_id], device=x.device)
            mod_emb = self.modality_embedding(mod_id)
            x = x + mod_emb.unsqueeze(1)
        x = self.self_attention(x)
        x = self.pool(x).squeeze(dim=1)  # (B, 512)
        return x

    def extract_tokens(self, x):
        """backbone + MSPE + self_attention 까지 수행 (pool 없음).
        반환: tokens (B, T, 512), unimodal_logits (B, n_classes)
        - tokens: bottleneck fusion 입력용 temporal 시퀀스
        - unimodal_logits: hybrid 앙상블 경로용 (pool->classifier, forward와 동일)
        """
        shape = x.shape
        x = x.view(-1, self.in_planes, x.shape[-2], x.shape[-1])
        n_frames_actual = shape[1] // self.in_planes
        if hasattr(self.backbone, 'update_n_frames'):
            self.backbone.update_n_frames(n_frames_actual)
        x = self.backbone(x)
        x = x.view(shape[0], shape[1] // self.in_planes, -1)
        if self.use_mspe:
            mod_id = torch.tensor([self.modality_id], device=x.device)
            mod_emb = self.modality_embedding(mod_id)
            x = x + mod_emb.unsqueeze(1)
        tokens = self.self_attention(x)              # (B, T, 512)
        pooled = self.pool(tokens).squeeze(dim=1)    # (B, 512)
        unimodal_logits = self.classifier(pooled)    # (B, n_classes)
        return tokens, unimodal_logits

def GestureTransoformer(backbone: str="resnet", in_planes: int=3, n_classes: int=25, **kwargs):
    if backbone not in backbone_dict:
        raise NotImplementedError("Backbone type: [{}] is not implemented.".format(backbone))
    model = _GestureTransformer(backbone_dict[backbone], in_planes, n_classes, **kwargs)
    return model