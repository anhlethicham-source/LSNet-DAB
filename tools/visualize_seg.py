"""
Visualize segmentation results with error maps (TP/FP/FN/TN) in black & white.

Usage:
    python tools/visualize_seg.py \
        configs/sem_fpn/fpn_lsnet_isic_2018.py \
        work_dirs/fpn_lsnet_isic_focal_dice/latest.pth \
        --img-dir /mnt/d/imagenet/data/images/test \
        --ann-dir /mnt/d/imagenet_processed/masks/test \
        --out-dir results/vis \
        --num-samples 20
"""

import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmcv import Config


# ── Error-map pixel values ──────────────────────────────────────────────
# TP = white (255), FP = light gray (180), FN = dark gray (80), TN = black (0)
TP_VAL = 255
FP_VAL = 180
FN_VAL = 80
TN_VAL = 0


def make_error_map(pred_bin, gt_bin):
    h, w = gt_bin.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[(pred_bin == 1) & (gt_bin == 1)] = TP_VAL   # TP — white
    canvas[(pred_bin == 1) & (gt_bin == 0)] = FP_VAL   # FP — light gray
    canvas[(pred_bin == 0) & (gt_bin == 1)] = FN_VAL   # FN — dark gray
    canvas[(pred_bin == 0) & (gt_bin == 0)] = TN_VAL   # TN — black
    return canvas


def compute_metrics(pred_bin, gt_bin):
    eps = 1e-6
    TP = int(((pred_bin == 1) & (gt_bin == 1)).sum())
    FP = int(((pred_bin == 1) & (gt_bin == 0)).sum())
    FN = int(((pred_bin == 0) & (gt_bin == 1)).sum())
    TN = int(((pred_bin == 0) & (gt_bin == 0)).sum())

    dice        = (2 * TP) / (2 * TP + FP + FN + eps)
    iou         = TP / (TP + FP + FN + eps)
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    precision   = TP / (TP + FP + eps)
    f1          = dice
    return dict(Dice=dice, IoU=iou, Sensitivity=sensitivity,
                Specificity=specificity, Precision=precision, F1=f1)


def add_legend(canvas):
    """Draw legend for TP/FP/FN/TN."""
    legend = [
        (TP_VAL, "TP (hit)"),
        (FP_VAL, "FP (false alarm)"),
        (FN_VAL, "FN (missed)"),
        (TN_VAL, "TN (background)"),
    ]
    x, y = 8, 18
    for val, label in legend:
        text_color = 0 if val > 120 else 255
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
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Build model ─────────────────────────────────────────────────────
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    model.eval()

    # ── Collect image / mask pairs ───────────────────────────────────────
    img_files = sorted([
        f for f in os.listdir(args.img_dir) if f.endswith('.jpg')
    ])[:args.num_samples]

    all_metrics = []

    for img_name in img_files:
        img_path = osp.join(args.img_dir, img_name)
        stem      = osp.splitext(img_name)[0]            # e.g. ISIC_0000001
        mask_name = stem + '_segmentation.png'
        mask_path = osp.join(args.ann_dir, mask_name)

        if not osp.exists(mask_path):
            print(f"[SKIP] mask not found: {mask_path}")
            continue

        # ── Inference ───────────────────────────────────────────────────
        result   = inference_segmentor(model, img_path)   # list[ndarray H×W]
        pred_seg = result[0].astype(np.uint8)             # class indices

        pred_bin = (pred_seg == 1).astype(np.uint8)
        gt_raw   = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_bin   = (gt_raw > 127).astype(np.uint8)

        # Resize gt to match prediction if needed
        if gt_bin.shape != pred_bin.shape:
            gt_bin = cv2.resize(gt_bin, (pred_bin.shape[1], pred_bin.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

        # ── Error map ───────────────────────────────────────────────────
        error_map = make_error_map(pred_bin, gt_bin)
        error_map = add_legend(error_map)

        # ── Per-image metrics ────────────────────────────────────────────
        m = compute_metrics(pred_bin, gt_bin)
        all_metrics.append(m)

        metric_str = (f"Dice={m['Dice']:.3f}  IoU={m['IoU']:.3f}  "
                      f"Sens={m['Sensitivity']:.3f}  Spec={m['Specificity']:.3f}  "
                      f"F1={m['F1']:.3f}")
        print(f"{stem}: {metric_str}")

        # ── Build side-by-side panel ─────────────────────────────────────
        orig      = cv2.imread(img_path)
        orig      = cv2.resize(orig, (256, 256))
        gt_vis    = (gt_bin * 255).astype(np.uint8)
        pred_vis  = (pred_bin * 255).astype(np.uint8)
        error_vis = cv2.resize(error_map, (256, 256))

        # Convert grayscale to BGR for stacking
        gt_bgr    = cv2.cvtColor(gt_vis,   cv2.COLOR_GRAY2BGR)
        pred_bgr  = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
        error_bgr = cv2.cvtColor(error_vis, cv2.COLOR_GRAY2BGR)

        # Add column titles
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

        # Add metric bar at bottom
        bar = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, metric_str, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.vstack([panel, bar])

        save_path = osp.join(args.out_dir, f"{stem}_vis.png")
        cv2.imwrite(save_path, panel)

    # ── Aggregate metrics ────────────────────────────────────────────────
    if all_metrics:
        print("\n" + "=" * 50)
        print(f"{'Metric':<15} {'Mean':>8} {'Std':>8}")
        print("-" * 35)
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            print(f"{key:<15} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}")
        print("=" * 50)
        print(f"\nSaved {len(all_metrics)} visualizations to: {args.out_dir}")


if __name__ == '__main__':
    main()
