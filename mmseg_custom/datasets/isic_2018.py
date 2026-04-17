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

        TP = FP = FN = TN = 0

        # When test.py runs with --eval, MMSeg uses pre_eval=True and each
        # result is a tuple (area_intersect, area_union, area_pred, area_label)
        # where every tensor has shape (num_classes,).
        if results and isinstance(results[0], tuple):
            for area_intersect, area_union, area_pred, area_label in results:
                # class index 1 = lesion
                tp = int(area_intersect[1])
                fp = int(area_pred[1]) - tp
                fn = int(area_label[1]) - tp
                tn = int(area_intersect[0])
                TP += tp; FP += fp; FN += fn; TN += tn
        else:
            # Raw prediction maps path (single-GPU / no pre_eval)
            gt_seg_maps = self.get_gt_seg_maps()
            for pred, gt in zip(results, gt_seg_maps):
                pred = np.array(pred)
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
        f1          =  (2 * precision * recall) / (precision + recall)

        eval_results = dict(
            Dice=round(dice, 4),
            IoU=round(iou, 4),
            mIoU=round(iou, 4),   # alias so training runner can track/save_best
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
