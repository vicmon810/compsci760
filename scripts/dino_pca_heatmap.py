import argparse
import numpy as np
import cv2
import torch

from models.dino_landmark import DinoBackbone
from sklearn.decomposition import PCA

def to_tensor(img: np.ndarray, size: int = 224) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return t.unsqueeze(0)

def pca_heatmap(feats: torch.Tensor, tokens_only: bool = True):
    """
    feats: [B,T,C] token features from ViT (preferred). Use PCA over channel to get 1D map per token.
    Returns: [H_p,W_p] heatmap normalized to [0,1]; assumes square patch grid.
    """
    if feats.dim() == 2:
        # fallback: no token dimension
        v = feats.detach().cpu().numpy()
        v = (v - v.mean()) / (v.std() + 1e-6)
        m = np.abs(v)
        return (m - m.min()) / (m.max() - m.min() + 1e-6)
    x = feats[0].detach().cpu().numpy()  # [T,C]
    # discard CLS if present
    if tokens_only and x.shape[0] in (197, 577):  # 14x14 + 1, or 24x24 + 1
        x = x[1:]
    T, C = x.shape
    grid = int(np.sqrt(T))
    x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)
    pca = PCA(n_components=1, svd_solver="auto", random_state=0)
    c1 = pca.fit_transform(x)  # [T,1]
    c1 = np.abs(c1.reshape(grid, grid))
    c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-6)
    # upsample to image size
    return c1

def overlay_heatmap(img_bgr: np.ndarray, map01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    map_resized = cv2.resize(map01, (W, H), interpolation=cv2.INTER_CUBIC)
    heat = (map_resized * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = cv2.addWeighted(heat, alpha, img_bgr, 1 - alpha, 0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--backbone", type=str, default="vit_small_patch16_224.dino")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)

    bb = DinoBackbone(name=args.backbone, pretrained=True)
    x = to_tensor(img, size=224)
    with torch.no_grad():
        feats = bb(x)  # [1,T,C]

    h = pca_heatmap(feats, tokens_only=True)
    vis = overlay_heatmap(img, h, alpha=0.45)
    cv2.imwrite(args.out, vis)
    print(f"[OK] saved heatmap to {args.out}")

if __name__ == "__main__":
    main()
