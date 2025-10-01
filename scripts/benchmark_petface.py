import argparse
import os
import csv
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root")
    ap.add_argument("--ann", required=True, help="CSV annotations")
    args = ap.parse_args()

    n = 0
    with open(args.ann, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: 
                continue
            img_path = row[0].strip()
            full = img_path if os.path.isabs(img_path) else os.path.join(args.root, img_path)
            if not os.path.exists(full):
                print(f"[MISS] {full}")
            else:
                im = cv2.imread(full)
                if im is None:
                    print(f"[ERR] cannot read {full}")
                else:
                    n += 1
                    if n <= 3:
                        print(f"[OK] {full}  shape={im.shape}")
    print(f"[DONE] checked {n} images.")

if __name__ == "__main__":
    main()
