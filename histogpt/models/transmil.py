import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    """
    Transformer Layer with Nystrom Attention
    """
    def __init__(self, dim=512, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    """
    Pyramid Position Encoding Generator
    """
    def __init__(self, dim=512):
        super(PPEG, self).__init__()

        self.proj0 = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):

        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)

        x = self.proj0(cnn_feat)
        x = x + cnn_feat
        x = x + self.proj1(cnn_feat)
        x = x + self.proj2(cnn_feat)

        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)

        return x


class TransMIL(nn.Module):
    """
    Transformer for Correlated Multiple Instance Learning
    """
    def __init__(self, dim_features, dim_model, num_classes):
        super(TransMIL, self).__init__()

        self.pos_layer = PPEG(dim_model)
        self._fc1 = nn.Sequential(nn.Linear(dim_features, dim_model), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim_model)
        self.layer2 = TransLayer(dim_model)
        self.norm = nn.LayerNorm(dim_model)
        self._fc2 = nn.Linear(dim_model, num_classes)

    def forward(self, data):

        h = data.float()
        h = self._fc1(h)

        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(data.device)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)
        h = self.norm(h)[:, 0]

        return self._fc2(h)
