# PetFace + DINO Landmark Estimation (MVP)

pipeline to:
- load PetFace-style landmark data,
- train a **DINO-backed** landmark regressor,
- generate **PCA heatmaps** from DINO features,
- leave a clean **hook** to add an LLM-driven ROI selector later.

> Tested only as *reference code*. You may need to tweak path/format to your copy of PetFace.


## Quickstart

```bash
# 0) (Optional) create env
conda create -n petface python=3.10 -y && conda activate petface

# 1) install deps
pip install -r requirements.txt

# 2) sanity check your dataset structure/annotations
python scripts/benchmark_petface.py --root /path/to/PetFace --ann /path/to/annotations.csv

# 3) train (coordinate regression)
python train_landmark.py   --cfg configs/default.yaml   --data.root /path/to/PetFace   --data.ann /path/to/annotations.csv

# 4) generate a PCA heatmap for a single image
python scripts/dino_pca_heatmap.py --image /path/to/img.jpg --out out_heatmap.jpg
```

## Expected dataset format (CSV, default)

The loader expects a CSV with columns:
- `image_path`: path relative to `data.root` or absolute.
- `bbox`: optional; format `x1,y1,x2,y2` (pixels).
- `landmarks`: a semicolon-joined list of `x,y` pairs in *pixels*, e.g. `x1,y1;x2,y2;...;xK,yK`

Example row:
```
images/dog_0001.jpg, 50,60,220,240, 12,34; 45,80; 90,120; 110,160
```
Spaces are ignored. See `scripts/benchmark_petface.py` for a validator.

## Switching to COCO JSON keypoints

If your PetFace copy is in COCO format, pass `--data.coco_json /path/file.json`.
The dataset will ignore the CSV and parse `keypoints` arrays per annotation.

## DINO backbone

We load DINO via `timm` (e.g., `vit_small_patch16_224.dino` or `vit_base_patch16_224.dino`). 
If unavailable, fallback to a ViT backbone (AugReg) and warn.

## LLM-driven ROI (hook)

See `utils/roi.py` (stub). It currently selects ROI by thresholding the DINO PCA heatmap.
To plug an LLM, replace `select_regions()` with a call to your LLM policy.

## License

This scaffold is for **educational/research** purposes. Check your dataset's license before redistribution.
# compsci760
