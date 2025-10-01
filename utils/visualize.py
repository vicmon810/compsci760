from typing import Tuple
import numpy as np
import cv2
import torch

def denorm_landmarks_on_image(img: np.ndarray, landmarks: torch.Tensor) -> np.ndarray:
    """
    Draw normalized landmarks back on image (assumes image is HxWxC RGB).
    """
    out = img.copy()
    H, W = out.shape[:2]
    pts = landmarks.detach().cpu().numpy()
    pts[:,0] = np.clip(pts[:,0] * W, 0, W-1)
    pts[:,1] = np.clip(pts[:,1] * H, 0, H-1)
    for (x, y) in pts.astype(int):
        cv2.circle(out, (x, y), 2, (0,255,0), -1)
    return out
