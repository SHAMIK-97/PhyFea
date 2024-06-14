from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import *
from mmseg.ops import resize

from .segformer_head import SegFormerHead
from ..builder import HEADS


def _get_relu(name: str) -> nn.Sequential:
    container = nn.Sequential()
    relu = nn.ReLU()
    container.add_module(f'{name}_relu', relu)

    return container


def _max_pool2D(name: str) -> nn.Sequential:
    container = nn.Sequential()
    pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
    container.add_module(f'{name}_maxpool_2d', pool1)
    container.add_module(f'{name}_maxpool_2d_pad_1', nn.ConstantPad2d(1, 1))
    return container


def _avg_pool2D(name: str) -> nn.Sequential:
    container = nn.Sequential()
    pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=1)
    container.add_module(f'{name}_maxpool_2d', pool1)
    container.add_module(f'{name}_maxpool_2d_pad_1', nn.ConstantPad2d(1, 1))
    return container


@HEADS.register_module()
class PhysicsFormer(SegFormerHead):

    def __init__(self,iterations=2,image_size=(1024,1024), **kwargs):

        super(PhysicsFormer, self).__init__(input_transform='multiple_select', **kwargs)
        self.classes = self.num_classes
        self.relu = _get_relu('relu')
        self.maxpool_1 = _max_pool2D("maxpool_1")
        self.avgpool_1 = _avg_pool2D("avgpool_1")
        self.pad_1 = nn.ConstantPad2d(1, 1)
        self.inference = super().set_inference_parent
        self.T = iterations
        self.image_size = image_size

    def opening(self, x):
        relu = x
        for iteration in range(self.T):
            x1 = self.maxpool_1(x)
            x = torch.matmul(x1, relu)

        return x

    def _rectification(self, original, dilated):

        offset = torch.sub(dilated, original, alpha=1)
        offset_mean = torch.mean(offset, dim=2, keepdim=True)
        offset_diff = torch.sub(offset, offset_mean, alpha=1)
        offset_relu = self.relu(offset_diff)
        final_dilation = torch.add(original, offset_relu, alpha=1)
        return final_dilation

    def selective_dilation(self, x):
        relu = x
        for iteration in range(self.T):
            x1 = self.avgpool_1(x)
            x2 = torch.matmul(x1, relu)
            x = self._rectification(x, x2)
        return x

    def final_operation(self, original, mode='opening'):

        if mode == 'opening':
            final_concatenated_opened = torch.mul(original, -1)
            final_concatenated_opening = self.pad_1(original)
            operated = self.opening(final_concatenated_opening)
            operated_normalized = F.normalize(operated)
            x = torch.matmul(operated_normalized, final_concatenated_opened)
            subtracted = torch.sub(final_concatenated_opened, x, alpha=1)
        else:
            final_concatenated_dilation = self.pad_1(original)
            operated = self.selective_dilation(final_concatenated_dilation)
            operated_normalized = F.normalize(operated)
            x = torch.matmul(operated_normalized, final_concatenated_dilation)
            subtracted = torch.sub(final_concatenated_dilation, x, alpha=1)

        l1_norm = torch.norm(subtracted, p=1)
        return l1_norm

    def forward(self, inputs):

        if self.inference:
            return super().forward(inputs)
        if not self.inference:
            logits = super().forward(inputs)
            logits_upscaled = resize(
                logits,
                size=self.image_size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            upscaled_softmax = logits_upscaled.softmax(dim=1)
            tensor_list = []
            perm = permutations(range(self.num_classes), 2)
            for i in perm:
                concatenated_tensor = torch.cat(
                    (upscaled_softmax[:, i[0]:i[0] + 1, ::], upscaled_softmax[:, i[1]:i[1] + 1, ::]), dim=1)
                logits_mean = torch.mean(concatenated_tensor, dim=1, keepdim=True)
                logits_sub = torch.sub(concatenated_tensor, logits_mean, alpha=1)
                concat_relu = self.relu(logits_sub)
                tensor_list.append(concat_relu)

            final_concatenated = torch.cat(tensor_list, dim=1)

            norm_opened = self.final_operation(final_concatenated)
            norm_dilated = self.final_operation(final_concatenated, mode='dilation')

            final_norm = torch.abs(norm_opened - norm_dilated)
            return final_norm, logits
