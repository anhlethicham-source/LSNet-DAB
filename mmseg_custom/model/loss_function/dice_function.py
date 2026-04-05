import torch
import torch.nn as nn
from mmseg.registry import MODELS 

@MODELS.register_module() 
class MyDiceLoss(nn.Module):
    def __init__(self, use_sigmoid=True, loss_weight=1.0, loss_name='loss_dice'):
        super(MyDiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        if self.use_sigmoid:
            pred = torch.sigmoid(pred)
        
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name
