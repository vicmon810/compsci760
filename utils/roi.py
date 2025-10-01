# Placeholder for LLM-driven ROI selection.
# Currently uses simple thresholding on PCA heatmap.
import numpy as np
import cv2

def select_regions(heatmap: np.ndarray, thr: float = 0.6, min_area: int = 50):
    """
    Args:
        heatmap: float32 [H,W] in [0,1]
    Returns:
        list of (x1,y1,x2,y2) boxes where heatmap exceeds threshold.
    """
    m = (heatmap >= thr).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= min_area:
            boxes.append((x,y,x+w,y+h))
    return boxes
