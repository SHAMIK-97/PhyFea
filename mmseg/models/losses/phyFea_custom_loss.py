import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)


@LOSSES.register_module()
class CustomLoss(CrossEntropyLoss):

    def __init__(self, alpha=0.0001):
        super(CustomLoss, self).__init__()
        self.alpha = alpha

    def forward(self,
                cls_score,
                label,
                l1_norms,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        cross_entropy_loss = super().forward(cls_score,label,weight,avg_factor,reduction_override,**kwargs)
        total_loss = cross_entropy_loss + self.alpha*l1_norms
        return total_loss

