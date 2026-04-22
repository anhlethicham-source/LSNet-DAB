import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES


# --- MarginalDice (ResMamba-ULite) -----------------------------------

def _dilate(mask, kernel_size=9):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=torch.float32)
    out = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)
    return torch.clamp(out, 0, 1)


def _erode(mask, kernel_size=9):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=torch.float32) / (kernel_size ** 2)
    out = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)
    return torch.floor(out + 1e-2)


def _margin_weight(mask, weight_in=3, weight_out=1, weight_margin=6, kernel_size=9):
    dilated = _dilate(mask, kernel_size)
    eroded = _erode(mask, kernel_size)
    return (dilated - eroded) * weight_margin + eroded * weight_in + (1 - dilated) * weight_out


@LOSSES.register_module()
class MarginalDiceLoss(nn.Module):
    """Spatially-weighted Dice loss from ResMamba-ULite.

    Higher penalty near object boundaries (margin zone) via morphological
    dilation/erosion on the GT mask. Receives raw logits for y_pred.
    """

    def __init__(self, weight_in=3, weight_out=1, weight_margin=6,
                 kernel_size=9, smooth=1e-3,
                 loss_name='loss_marginal_dice', loss_weight=1.0, **kwargs):
        super().__init__()
        self.weight_in = weight_in
        self.weight_out = weight_out
        self.weight_margin = weight_margin
        self.kernel_size = kernel_size
        self.smooth = smooth
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        target = target.float()
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # 1. Tính weight map TRƯỚC (từ ground truth, không cần gradient)
        with torch.no_grad():
            weight = _margin_weight(target, self.weight_in, self.weight_out,
                                    self.weight_margin, self.kernel_size)

        # 2. Tính probability
        prob = torch.sigmoid(pred)

        # 3. Weighted Dice THỰC SỰ per-pixel, per-sample
        # sum theo (C, H, W), giữ chiều batch
        w_intersection = (weight * prob * target).sum(dim=(1, 2, 3))  # (B,)
        w_pred_sum = (weight * prob).sum(dim=(1, 2, 3))  # (B,)
        w_target_sum = (weight * target).sum(dim=(1, 2, 3))  # (B,)

        dice = (2.0 * w_intersection + self.smooth) / (
                w_pred_sum + w_target_sum + self.smooth
        )  # (B,)

        loss = (1.0 - dice).mean()  # scalar
        return loss
    @property
    def loss_name(self):
        return self._loss_name


@LOSSES.register_module()
class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, wf=0.5, wd=0.5, smooth=1.0, loss_name='loss_focal_dice', use_sigmoid=None, loss_weight=1.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.wf = wf
        self.wd = wd
        self.smooth = smooth
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        target = target.float()
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Focal Loss
        prob = torch.sigmoid(pred)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = prob * target + (1 - prob) * (1 - target)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma * bce).mean()

        # Dice Loss
        prob_flat = prob.contiguous().view(prob.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)
        intersection = (prob_flat * target_flat).sum(dim=1)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) /
                           (prob_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)).mean()

        return self.wf * focal_loss + self.wd * dice_loss

    @property
    def loss_name(self):
        return self._loss_name


@LOSSES.register_module()
class MyBCEDiceLoss(nn.Module):
    def __init__(
        self,
        wb=0.5,
        wd=0.5,
        smooth=1.0,
        pos_weight=None,
        loss_name='loss_bce_dice'
    ):
        super().__init__()
        self.wb = wb
        self.wd = wd
        self.smooth = smooth
        self._loss_name = loss_name

        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, **kwargs):
        # pred: [N, 1, H, W], target: [N, H, W] or [N, 1, H, W]
        target = target.float()
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # BCE on logits
        bce_loss = self.bce(pred, target)

        # Dice on probabilities
        prob = torch.sigmoid(pred)
        prob = prob.contiguous().view(prob.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (prob * target).sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (
            prob.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()

        return self.wb * bce_loss + self.wd * dice_loss

    @property
    def loss_name(self):
        return self._loss_name


# --- CombinedSegLoss: Focal + Dice + MarginalDice --------------------

@LOSSES.register_module()
class CombinedSegLoss(nn.Module):
    """Focal + standard Dice + MarginalDice (boundary-aware).

    Focal      → xử lý class imbalance, học toàn bộ ảnh
    Dice       → tối ưu trực tiếp overlap (DSC)
    MarginalDice → phạt nặng vùng viền để cải thiện IoU
    """

    def __init__(self,
                 wf=0.4, wd=0.3, wm=0.5,
                 gamma=2.0, alpha=0.25,
                 weight_in=3, weight_out=1, weight_margin=6, kernel_size=9,
                 smooth=1.0,
                 loss_name='loss_combined', loss_weight=1.0, **kwargs):
        super().__init__()
        self.wf = wf
        self.wd = wd
        self.wm = wm
        self.gamma = gamma
        self.alpha = alpha
        self.weight_in = weight_in
        self.weight_out = weight_out
        self.weight_margin = weight_margin
        self.kernel_size = kernel_size
        self.smooth = smooth
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        target = target.float()
        if target.dim() == 3:
            target = target.unsqueeze(1)

        prob = torch.sigmoid(pred)

        # 1. Focal loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = prob * target + (1 - prob) * (1 - target)
        focal = (self.alpha * (1 - p_t) ** self.gamma * bce).mean()

        # 2. Standard Dice loss
        prob_flat = prob.contiguous().view(prob.size(0), -1)
        tgt_flat  = target.contiguous().view(target.size(0), -1)
        inter = (prob_flat * tgt_flat).sum(dim=1)
        dice  = 1.0 - ((2.0 * inter + self.smooth) /
                       (prob_flat.sum(dim=1) + tgt_flat.sum(dim=1) + self.smooth)).mean()

        # 3. MarginalDice — boundary-aware, weight map không cần gradient
        with torch.no_grad():
            weight = _margin_weight(target, self.weight_in, self.weight_out,
                                    self.weight_margin, self.kernel_size)
        w_inter   = (weight * prob * target).sum(dim=(1, 2, 3))
        w_pred    = (weight * prob).sum(dim=(1, 2, 3))
        w_tgt     = (weight * target).sum(dim=(1, 2, 3))
        marginal  = (1.0 - (2.0 * w_inter + 1e-3) /
                     (w_pred + w_tgt + 1e-3)).mean()

        return self.wf * focal + self.wd * dice + self.wm * marginal

    @property
    def loss_name(self):
        return self._loss_name