from mmseg.datasets import CustomDataset, DATASETS
import numpy as np


@DATASETS.register_module(force=True)
class ISIC2018Dataset(CustomDataset):
    CLASSES = ('background', 'lesion')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='_segmentation.png', **kwargs)

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Compute Dice, IoU, Recall, Sensitivity, Specificity, F1, Precision."""
        gt_seg_maps = self.get_gt_seg_maps()

        TP = FP = FN = TN = 0
        for pred, gt in zip(results, gt_seg_maps):
            # pred is either a 2D array of class indices or probability map
            if pred.ndim == 3:
                pred = pred.argmax(axis=0)
            pred_bin = (pred == 1).astype(np.uint8)
            gt_bin   = (gt  == 1).astype(np.uint8)

            TP += int(((pred_bin == 1) & (gt_bin == 1)).sum())
            FP += int(((pred_bin == 1) & (gt_bin == 0)).sum())
            FN += int(((pred_bin == 0) & (gt_bin == 1)).sum())
            TN += int(((pred_bin == 0) & (gt_bin == 0)).sum())

        eps = 1e-6
        dice        = (2 * TP) / (2 * TP + FP + FN + eps)
        iou         = TP / (TP + FP + FN + eps)
        recall      = TP / (TP + FN + eps)      # = Sensitivity
        sensitivity = recall
        specificity = TN / (TN + FP + eps)
        precision   = TP / (TP + FP + eps)
        f1          = dice                       # F1 == Dice for binary

        eval_results = dict(
            Dice=round(dice, 4),
            IoU=round(iou, 4),
            Recall=round(recall, 4),
            Sensitivity=round(sensitivity, 4),
            Specificity=round(specificity, 4),
            Precision=round(precision, 4),
            F1=round(f1, 4),
        )

        # Pretty print
        header = f"\n{'Metric':<15} {'Value':>8}"
        sep    = "-" * 25
        lines  = [header, sep]
        for k, v in eval_results.items():
            lines.append(f"{k:<15} {v:>8.4f}")
        lines.append(sep)
        print("\n".join(lines))

        return eval_results
