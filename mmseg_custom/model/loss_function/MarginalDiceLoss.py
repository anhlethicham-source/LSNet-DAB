import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES

def dilatted(mask, kernel_size=9):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device).float()
    mask = F.conv2d(mask, kernel, padding="same")
    return torch.clip(mask, 0.0, 1.0)


def erosin(mask, kernel_size=9):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device) * (1.0 / kernel_size ** 2)
    mask = F.conv2d(mask, kernel, padding="same")
    return torch.floor(mask + 1e-2)


def marginweight(mask, weight_in=3, weight_out=1, weight_margin=6, kernel_size=9):
    mask_dilated = dilatted(mask, kernel_size)
    mask_erosin = erosin(mask, kernel_size)

    weight = (mask_dilated - mask_erosin) * weight_margin \
             + mask_erosin * weight_in \
             + (1.0 - mask_dilated) * weight_out
    return weight


def dice_score(pred, target, smooth=1e-3):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


@LOSSES.register_module()
class MarginalDiceLoss(nn.Module):
    """Marginal Dice Loss for MMSegmentation."""

    def __init__(self,
                 weight_in=3,
                 weight_out=1,
                 weight_margin=6,
                 kernel_size=9,
                 smooth=1e-3,
                 loss_weight=1.0,
                 loss_name='loss_marginal_dice',
                 **kwargs):
        super().__init__()
        self.weight_in = weight_in
        self.weight_out = weight_out
        self.weight_margin = weight_margin
        self.kernel_size = kernel_size
        self.smooth = smooth
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                ignore_index=255,
                **kwargs):
        """
        Args:
            cls_score (Tensor): Raw logits from model, shape (B, C, H, W).
            label (Tensor): Ground truth mask, shape (B, H, W) or (B, 1, H, W).
            ignore_index (int): Label index to ignore (typically 255).
        """
        # 1. Handle ignore_index (exclude region 255 from loss computation)
        valid_mask = (label != ignore_index).float()

        # Replace ignore_index pixels with 0 to prevent morphology errors
        clean_label = torch.where(label == ignore_index, torch.zeros_like(label), label)

        # 2. Reshape label dimensions to (B, 1, H, W)
        clean_label = clean_label.float()
        if clean_label.dim() == 3:
            clean_label = clean_label.unsqueeze(1)
            valid_mask = valid_mask.unsqueeze(1)

        # Ensure cls_score channels match label (binary segmentation)
        # If model sets num_classes=1, cls_score is already (B, 1, H, W)

        # 3. Compute Weight Map from Ground Truth
        with torch.no_grad():
            margin_w = marginweight(clean_label, self.weight_in, self.weight_out,
                                    self.weight_margin, self.kernel_size)
            # Do not penalize ignore_index regions
            margin_w = margin_w * valid_mask

            # 4. Compute pixel-wise BCE weighted by margin
        bce = F.binary_cross_entropy_with_logits(cls_score, clean_label, reduction='none')
        loss = (margin_w * bce * valid_mask).sum() / (valid_mask.sum() + 1e-5)

        # 5. Add global Dice to stabilize convergence
        dice = dice_score(cls_score, clean_label, self.smooth)
        loss = loss + (1.0 - dice)

        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
