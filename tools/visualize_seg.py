"""
Visualize segmentation results with error maps (TP/FP/FN/TN) and save
predicted masks in the same binary format as ISIC ground-truth masks
(0 = background, 255 = lesion).

Usage:
    python tools/visualize_seg.py \
        configs/sem_fpn/fpn_lsnet_isic_2018.py \
        work_dirs/fpn_lsnet_isic_focal_dice/latest.pth \
        --img-dir /mnt/d/imagenet/data/images/test \
        --ann-dir /mnt/d/imagenet_processed/masks/test \
        --out-dir results/vis \
        --num-samples 20

Outputs (inside --out-dir):
    panels/   — side-by-side image | GT | prediction | error-map
    masks/    — predicted mask PNGs (0/255, same format as GT)
"""

import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor


# ── Error-map pixel values ──────────────────────────────────────────────
# TP = white (255), FP = light gray (180), FN = dark gray (80), TN = black (0)
TP_VAL = 255
FP_VAL = 180
FN_VAL = 80
TN_VAL = 0


def make_error_map(pred_bin, gt_bin):
    h, w = gt_bin.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[(pred_bin == 1) & (gt_bin == 1)] = TP_VAL
    canvas[(pred_bin == 1) & (gt_bin == 0)] = FP_VAL
    canvas[(pred_bin == 0) & (gt_bin == 1)] = FN_VAL
    canvas[(pred_bin == 0) & (gt_bin == 0)] = TN_VAL
    return canvas


def compute_metrics(pred_bin, gt_bin):
    eps = 1e-6
    TP = int(((pred_bin == 1) & (gt_bin == 1)).sum())
    FP = int(((pred_bin == 1) & (gt_bin == 0)).sum())
    FN = int(((pred_bin == 0) & (gt_bin == 1)).sum())
    TN = int(((pred_bin == 0) & (gt_bin == 0)).sum())

    dice        = (2 * TP) / (2 * TP + FP + FN + eps)
    iou         = TP / (TP + FP + FN + eps)
    recall      = TP / (TP + FN + eps)      # = Sensitivity
    sensitivity = recall
    specificity = TN / (TN + FP + eps)
    precision   = TP / (TP + FP + eps)
    f1          = (2 * precision * recall) / (precision + recall + eps)

    return dict(Dice=dice, IoU=iou, Sensitivity=sensitivity,
                Specificity=specificity, Precision=precision, F1=f1)


def add_legend(canvas):
    legend = [
        (TP_VAL, "TP (hit)"),
        (FP_VAL, "FP (false alarm)"),
        (FN_VAL, "FN (missed)"),
        (TN_VAL, "TN (background)"),
    ]
    x, y = 8, 18
    for val, label in legend:
        cv2.rectangle(canvas, (x, y - 12), (x + 16, y + 2), int(val), -1)
        cv2.rectangle(canvas, (x, y - 12), (x + 16, y + 2), 128, 1)
        cv2.putText(canvas, label, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, 200, 1, cv2.LINE_AA)
        y += 20
    return canvas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--img-dir',     required=True)
    parser.add_argument('--ann-dir',     required=True)
    parser.add_argument('--out-dir',     default='results/vis')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--device',      default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()

    panel_dir = osp.join(args.out_dir, 'panels')
    mask_dir  = osp.join(args.out_dir, 'masks')
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(mask_dir,  exist_ok=True)

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model.eval()

    img_files = sorted([
        f for f in os.listdir(args.img_dir) if f.endswith('.jpg')
    ])[:args.num_samples]

    all_metrics = []

    for img_name in img_files:
        img_path  = osp.join(args.img_dir, img_name)
        stem      = osp.splitext(img_name)[0]          # e.g. ISIC_0000001
        mask_name = stem + '_segmentation.png'
        mask_path = osp.join(args.ann_dir, mask_name)

        if not osp.exists(mask_path):
            print(f"[SKIP] mask not found: {mask_path}")
            continue

        # ── Inference ───────────────────────────────────────────────────
        result   = inference_segmentor(model, img_path)
        pred_seg = result[0].astype(np.uint8)           # class indices H×W
        pred_bin = (pred_seg == 1).astype(np.uint8)

        gt_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt_raw > 127).astype(np.uint8)

        if gt_bin.shape != pred_bin.shape:
            gt_bin = cv2.resize(gt_bin, (pred_bin.shape[1], pred_bin.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

        # ── Save predicted mask — binary PNG matching GT format (0/255) ─
        pred_mask_png = (pred_bin * 255).astype(np.uint8)
        cv2.imwrite(osp.join(mask_dir, mask_name), pred_mask_png)

        # ── Error map ───────────────────────────────────────────────────
        error_map = make_error_map(pred_bin, gt_bin)
        error_map = add_legend(error_map)

        # ── Per-image metrics (same formula as isic_2018.py evaluate) ───
        m = compute_metrics(pred_bin, gt_bin)
        all_metrics.append(m)

        metric_str = (f"Dice={m['Dice']:.3f}  IoU={m['IoU']:.3f}  "
                      f"Sens={m['Sensitivity']:.3f}  Spec={m['Specificity']:.3f}  "
                      f"Prec={m['Precision']:.3f}  F1={m['F1']:.3f}")
        print(f"{stem}: {metric_str}")

        # ── Side-by-side panel ──────────────────────────────────────────
        orig      = cv2.imread(img_path)
        orig      = cv2.resize(orig, (256, 256))
        gt_vis    = (gt_bin * 255).astype(np.uint8)
        pred_vis  = pred_mask_png.copy()
        error_vis = cv2.resize(error_map, (256, 256))

        gt_bgr    = cv2.cvtColor(gt_vis,   cv2.COLOR_GRAY2BGR)
        pred_bgr  = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
        error_bgr = cv2.cvtColor(error_vis, cv2.COLOR_GRAY2BGR)

        def add_title(img, title):
            out = img.copy()
            cv2.putText(out, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)
            return out

        panel = np.hstack([
            add_title(orig,       "Image"),
            add_title(gt_bgr,     "Ground Truth"),
            add_title(pred_bgr,   "Prediction"),
            add_title(error_bgr,  "Error Map"),
        ])

        bar = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, metric_str, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.vstack([panel, bar])

        cv2.imwrite(osp.join(panel_dir, f"{stem}_vis.png"), panel)

    # ── Aggregate metrics ────────────────────────────────────────────────
    if all_metrics:
        print("\n" + "=" * 55)
        print(f"{'Metric':<15} {'Mean':>8} {'Std':>8}")
        print("-" * 35)
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            print(f"{key:<15} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}")
        print("=" * 55)
        print(f"\nPanels  → {panel_dir}")
        print(f"Masks   → {mask_dir}  (binary PNG, 0=bg / 255=lesion)")
        print(f"Saved {len(all_metrics)} samples.")


if __name__ == '__main__':
    main()
