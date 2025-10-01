# petface_dataset.py (drop-in replacement)

import os
import csv
import json
import math
import re
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def _parse_bbox_str(s: Optional[str]) -> Optional[Tuple[float,float,float,float]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    if len(parts) != 4:
        return None
    try:
        vals = [float(x) for x in parts]
    except:
        return None
    return tuple(vals)  # x1,y1,x2,y2

def _parse_landmarks_str(s: str) -> np.ndarray:
    # "x1,y1; x2,y2; ..."
    s = s.strip()
    if not s:
        raise ValueError("Empty landmark string")
    pairs = [p.strip() for p in s.split(";") if p.strip()]
    pts = []
    for pr in pairs:
        xy = [q.strip() for q in pr.replace(" ", "").split(",") if q.strip()]
        if len(xy) != 2:
            raise ValueError(f"Bad pair: {pr}")
        pts.append([float(xy[0]), float(xy[1])])
    if not pts:
        raise ValueError("No valid landmarks parsed from string")
    return np.array(pts, dtype=np.float32)  # [K,2]

def _parse_landmarks_flexible(row: List[str], start_idx: int) -> np.ndarray:
    """
    更宽容的解析器，支持：
      - 单元格形式 "x,y; x,y; ..."
      - 宽表: x1, y1, x2, y2, ...
      - COCO 三元组: x1, y1, v1, x2, y2, v2, ...（忽略 v）
      - 单元格 "x,y"（没有分号）
    """
    cell = row[start_idx].strip() if start_idx < len(row) else ""
    # 快速路径：分号分隔
    if ";" in cell:
        try:
            return _parse_landmarks_str(cell)
        except Exception:
            pass

    nums: List[float] = []
    for t in row[start_idx:]:
        t = (t or "").strip()
        if not t:
            continue
        # 像 "12,34" 这种
        if "," in t and ";" not in t:
            for p in [p.strip() for p in t.split(",") if p.strip()]:
                try:
                    nums.append(float(p))
                except:
                    pass
        else:
            try:
                nums.append(float(t))
            except:
                # 非数字忽略
                pass

    if len(nums) < 4:
        raise ValueError("Not enough numeric tokens for landmarks")

    # 优先按 COCO 三元组
    if len(nums) % 3 == 0 and (len(nums) // 3) >= 2:
        pts = []
        for i in range(0, len(nums), 3):
            x, y = nums[i], nums[i+1]  # 忽略 v
            pts.append([x, y])
        return np.array(pts, dtype=np.float32)

    # 否则按 XY 成对
    if len(nums) % 2 == 0:
        pts = []
        for i in range(0, len(nums), 2):
            pts.append([nums[i], nums[i+1]])
        return np.array(pts, dtype=np.float32)

    raise ValueError("Numeric tokens do not form XY pairs or COCO triples")

def _bbox_from_pts(pts: np.ndarray) -> Tuple[float,float,float,float]:
    x1, y1 = np.min(pts[:,0]), np.min(pts[:,1])
    x2, y2 = np.max(pts[:,0]), np.max(pts[:,1])
    return float(x1), float(y1), float(x2), float(y2)

def _landmarks_from_bbox(b, mode="5p"):
    # b: (x1, y1, x2, y2)
    x1, y1, x2, y2 = b
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if mode == "4p":
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:  # 5p: 四角 + 中心
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [cx, cy]]
    return np.array(pts, dtype=np.float32)

class PetFaceLandmarkDataset(Dataset):
    """
    CSV 支持以下模式：
      (A) image_path, landmarks_str
      (B) image_path, bbox_str, landmarks_str
      (C) image_path, x1, y1, x2, y2, ...
      (D) image_path, x1, y1, v1, x2, y2, v2, ...
      (E) image_path, bbox_str, x1, y1, x2, y2, ...
    也支持 COCO JSON（coco_json）。
    """
    def __init__(self,
                 root: str,
                 ann_csv: Optional[str] = None,
                 coco_json: Optional[str] = None,
                 input_size: int = 224,
                 crop: bool = True,
                 normalize: bool = True,
                 augment: bool = True):
        super().__init__()
        self.root = root
        self.input_size = int(input_size)
        self.crop = crop
        self.normalize = normalize
        self.samples = []

        if coco_json:
            self._load_coco(coco_json)
        else:
            if not ann_csv:
                raise ValueError("CSV annotations path must be provided if not using COCO JSON.")
            self._load_csv(ann_csv)

        self.augment = augment
        self.tf_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.input_size, self.input_size)),
        ])
        self.tf_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.input_size, self.input_size)),
        ])

    def _load_csv(self, ann_csv: str):
        with open(ann_csv, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                image_path = (row[0] or "").strip().rstrip("|")
                lower = image_path.lower()
                # 跳过表头/非图片行
                if lower in ("image", "image_path", "image_name") or \
                   not re.search(r"\.(jpg|jpeg|png|bmp|webp)$", lower):
                    continue

                bbox = None
                landmarks = None
                try:
                    # 你的 CSV: image_name, description, x1(min), x2(max), y1(min), y2(max)
                    if len(row) >= 6 and row[1] and not row[1].strip().replace('.', '', 1).isdigit():
                        # 有描述列 -> 解析第 3~6 列为 bbox
                        nums = []
                        for t in row[2:6]:
                            t = (t or "").strip()
                            if t == "":
                                continue
                            nums.append(float(t))
                        if len(nums) == 4:
                            x1, x2, y1, y2 = nums  # 先按你的列顺序取
                            bbox = (x1, y1, x2, y2)  # 转成 x1,y1,x2,y2
                        else:
                            # 如果 bbox 不完整，后面用整图替代
                            bbox = None
                        # landmarks 留空，后面会用 bbox 生成伪关键点
                        landmarks = None
                    else:
                        # 一般情况：尝试 "bbox_str + landmarks" / 纯 landmarks / 宽表
                        maybe_bbox = _parse_bbox_str(row[1]) if len(row) >= 2 else None
                        if maybe_bbox is not None:
                            bbox = maybe_bbox
                            try:
                                landmarks = _parse_landmarks_flexible(row, 2)
                            except Exception:
                                landmarks = None
                        else:
                            landmarks = _parse_landmarks_flexible(row, 1)
                except Exception as e:
                    raise ValueError(f"Row parse failed at image={image_path}: {e}")

                self.samples.append({
                    "img": image_path,
                    "bbox": bbox,
                    "landmarks": landmarks
                })

    def _load_coco(self, coco_json_path: str):
        with open(coco_json_path, "r") as f:
            data = json.load(f)
        id2img = {im["id"]: im for im in data["images"]}
        for ann in data["annotations"]:
            if "keypoints" not in ann:
                continue
            kp = ann["keypoints"]
            pts = []
            for i in range(0, len(kp), 3):
                x, y = float(kp[i]), float(kp[i+1])
                pts.append([x, y])
            pts = np.array(pts, dtype=np.float32)
            image_path = id2img[ann["image_id"]]["file_name"]
            if "bbox" in ann:
                x, y, w, h = ann["bbox"]
                bbox_xyxy = (x, y, x+w, y+h)
            else:
                bbox_xyxy = _bbox_from_pts(pts)
            self.samples.append({
                "img": image_path,
                "bbox": bbox_xyxy,
                "landmarks": pts
            })

    def __len__(self):
        return len(self.samples)

    def _crop_and_resize(self, img, pts, bbox):
        H, W = img.shape[:2]
        if bbox is None:
            bbox = _bbox_from_pts(pts)
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(math.floor(x1)))
        y1 = max(0, int(math.floor(y1)))
        x2 = min(W-1, int(math.ceil(x2)))
        y2 = min(H-1, int(math.ceil(y2)))
        roi = img[y1:y2+1, x1:x2+1, :].copy()
        pts_adj = pts.copy()
        pts_adj[:,0] -= x1
        pts_adj[:,1] -= y1
        roi_resized = cv2.resize(roi, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        sx = self.input_size / max(1, roi.shape[1])
        sy = self.input_size / max(1, roi.shape[0])
        pts_resized = pts_adj.copy()
        pts_resized[:,0] *= sx; pts_resized[:,1] *= sy
        return roi_resized, pts_resized

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_path = rec["img"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root, img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 若无关键点，用 bbox 生成 5 个伪关键点（四角+中心）
        pts = rec["landmarks"]
        if pts is None:
            b = rec["bbox"]
            if b is None:
                H, W = img.shape[:2]
                b = (0.0, 0.0, float(W-1), float(H-1))
            pts = _landmarks_from_bbox(b, mode="5p")
        pts = pts.astype(np.float32)

        if self.crop:
            img, pts = self._crop_and_resize(img, pts, rec["bbox"])
            tensor = transforms.ToTensor()(img)
        else:
            tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.input_size, self.input_size)),
            ])(img)
            H, W = img.shape[:2]
            sx = self.input_size / W; sy = self.input_size / H
            pts = pts.copy(); pts[:,0] *= sx; pts[:,1] *= sy

        target = pts.copy()
        if self.normalize:
            target[:,0] /= tensor.shape[2]  # width
            target[:,1] /= tensor.shape[1]  # height

        return tensor, torch.from_numpy(target).float()
