import os
import argparse
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from petface_dataset import PetFaceLandmarkDataset
from models.dino_landmark import DinoBackbone, LandmarkRegressor
from utils.metrics import compute_nme
from utils.visualize import denorm_landmarks_on_image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="configs/default.yaml")
    p.add_argument("--data.root", dest="data_root", type=str, required=False)
    p.add_argument("--data.ann", dest="data_ann", type=str, required=False)
    p.add_argument("--data.coco_json", dest="data_coco_json", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_landmarks", type=int, default=None)
    p.add_argument("--save_dir", type=str, default="runs/petface")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--backbone", type=str, default=None)  # e.g. vit_small_patch16_224.dino
    return p.parse_args()

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def deep_update(d: Dict[str, Any], u: Dict[str, Any]):
    for k, v in u.items():
        if isinstance(v, dict) and k in d:
            deep_update(d[k], v)
        else:
            d[k] = v

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)

    # CLI overrides
    if args.data_root: cfg["data"]["root"] = args.data_root
    if args.data_ann: cfg["data"]["ann"] = args.data_ann
    if args.data_coco_json: cfg["data"]["coco_json"] = args.data_coco_json
    if args.epochs: cfg["train"]["epochs"] = args.epochs
    if args.batch: cfg["train"]["batch_size"] = args.batch
    if args.lr: cfg["train"]["lr"] = args.lr
    if args.num_landmarks is not None: cfg["model"]["num_landmarks"] = args.num_landmarks
    if args.backbone: cfg["model"]["backbone"] = args.backbone

    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets
    ds_train = PetFaceLandmarkDataset(
        root=cfg["data"]["root"],
        ann_csv=cfg["data"].get("ann"),
        coco_json=cfg["data"].get("coco_json"),
        input_size=cfg["data"]["input_size"],
        crop=cfg["data"]["crop"],
        normalize=True,
        augment=True
    )
    ds_val = PetFaceLandmarkDataset(
        root=cfg["data"]["root"],
        ann_csv=cfg["data"].get("ann_val", cfg["data"].get("ann")),
        coco_json=cfg["data"].get("coco_json_val"),
        input_size=cfg["data"]["input_size"],
        crop=cfg["data"]["crop"],
        normalize=True,
        augment=False
    )
    dl_train = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["workers"], pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["workers"], pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoBackbone(name=cfg["model"]["backbone"], pretrained=True)
    model = LandmarkRegressor(backbone, num_landmarks=cfg["model"]["num_landmarks"], use_cls=cfg["model"]["use_cls"]).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["wd"])
    criterion = nn.SmoothL1Loss(beta=0.05)  # robust for coords

    best_val = 1e9
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for imgs, targets in dl_train:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        # validate
        model.eval()
        val_loss = 0.0
        nme_sum = 0.0
        count = 0
        with torch.no_grad():
            for imgs, targets in dl_val:
                imgs = imgs.to(device)
                targets = targets.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * imgs.size(0)
                # NME in normalized coords (w=h=1); multiply by diag=sqrt(2) to get pixel-agnostic
                nme = compute_nme(preds, targets)  # averaged per batch
                nme_sum += nme * imgs.size(0)
                count += imgs.size(0)

        train_loss = total_loss / len(ds_train)
        val_loss = val_loss / len(ds_val)
        val_nme = nme_sum / count if count > 0 else 0.0

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_NME={val_nme:.4f}")

        # checkpoint by NME
        if val_nme < best_val:
            best_val = val_nme
            path = os.path.join(args.save_dir, f"best_epoch{epoch:03d}_nme{best_val:.4f}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, path)
            print(f"[CKPT] saved {path}")

if __name__ == "__main__":
    main()
