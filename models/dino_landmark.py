from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

class DinoBackbone(nn.Module):
    def __init__(self, name: str = "vit_small_patch16_224.dino", pretrained: bool = True, feat_dim: int = 384):
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        if timm is None:
            raise ImportError("timm is required for DINO backbones. Install timm.")
        try:
            self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
            self.feat_dim = self.backbone.num_features if hasattr(self.backbone, "num_features") else feat_dim
        except Exception as e:
            # fallback
            fallback = "vit_base_patch16_224.augreg_in21k"
            self.backbone = timm.create_model(fallback, pretrained=True, num_classes=0, global_pool="")
            self.feat_dim = self.backbone.num_features
            print(f"[WARN] Failed to load {name}. Fallback to {fallback}. Err: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect shape [B, C, H, W]; return token sequence or pooled?
        # Many timm ViTs return [B, tokens, C] via forward_features
        feats = self.backbone.forward_features(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        # feats could be [B, tokens, C] or [B, C]
        return feats

class LandmarkRegressor(nn.Module):
    """
    Simple coordinate regressor: use CLS token or mean pooled tokens.
    Predict K*2 normalized coords in [0,1].
    """
    def __init__(self, backbone: DinoBackbone, num_landmarks: int, use_cls: bool = True, dropout: float = 0.1):
        super().__init__()
        self.bb = backbone
        self.use_cls = use_cls
        self.num_landmarks = num_landmarks
        in_dim = backbone.feat_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, num_landmarks * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.bb(x)  # [B, T, C] or [B, C]
        if feats.dim() == 3:
            if self.use_cls:
                x = feats[:, 0]  # CLS token
            else:
                x = feats.mean(dim=1)  # mean pool tokens
        else:
            x = feats
        out = self.head(x)
        return out.view(x.size(0), self.num_landmarks, 2)
