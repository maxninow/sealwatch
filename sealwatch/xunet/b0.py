"""Implementation of EfficientNet B0.

Inspired by `timm` package.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F


class B0(nn.Module):
    """EfficientNet B0 implementation."""

    def __init__(self, num_channels: int = 3, num_classes: int = 2, no_stem_stride: bool = False):
        """Constructor.

        :param num_channels: number of input channels
        :param num_classes: number of output classes
        :param no_stem_stride: whether to subsample in the first layer
        """
        super().__init__()
        # stem layer
        stem_stride = 1 if no_stem_stride else 2
        self.conv_stem = nn.Conv2d(num_channels, 32, kernel_size=3, stride=stem_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU()

        # Define block structure: (in_channels, out_channels, num_blocks, stride, kernel_size, exp_ratio, se_ratio)
        block_args = [
            (32, 16, 1, 1, 3, 1, 1/4),
            (16, 24, 2, 2, 3, 6, 1/24),
            (24, 40, 2, 2, 5, 6, 1/24),
            (40, 80, 3, 2, 3, 6, 1/24),
            (80, 112, 3, 1, 5, 6, 1/24),
            (112, 192, 4, 2, 5, 6, 1/24),
            (192, 320, 1, 1, 3, 6, 1/24),
        ]

        blocks = []
        for in_chs, out_chs, num_blocks, stride, dw_kernel_size, exp_ratio, se_ratio in block_args:
            layers = []
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                if exp_ratio == 1.0:
                    layers.append(DepthwiseSeparableConv(in_chs, out_chs, stride=s, dw_kernel_size=dw_kernel_size))
                else:
                    layers.append(InvertedResidual(in_chs, out_chs, stride=s, dw_kernel_size=dw_kernel_size, exp_ratio=exp_ratio, se_ratio=se_ratio, use_se=True))
                in_chs = out_chs
            blocks.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*blocks)

        # Head
        self.conv_head = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.act2 = nn.SiLU()

        # FC
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        """Forward pass.

        :param x: input image
        """
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        #
        x = self.blocks(x)
        #
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        #
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution. Component of EfficientNet."""

    def __init__(self, in_chs: int, out_chs: int, dw_kernel_size: int = 3, stride: int = 1, pw_act: bool = False):
        """Constructor."""
        super(DepthwiseSeparableConv, self).__init__()
        padding = (dw_kernel_size - 1) // 2

        self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size=dw_kernel_size, stride=stride, padding=padding, groups=in_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.act1 = nn.SiLU()

        self.se = SqueezeExcite(in_chs, se_ratio=.25)

        self.conv_pw = nn.Conv2d(in_chs, out_chs, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs)
        self.act2 = nn.SiLU() if pw_act else nn.Identity()

    def forward(self, x):
        """Forward pass."""
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class InvertedResidual(nn.Module):
    """Inverted residual. Component of EfficientNet B0."""

    def __init__(self, in_chs: int, out_chs: int, stride: int = 1, dw_kernel_size: int = 3, exp_ratio: float = 1.0, se_ratio: float = .25, use_se: bool = False):
        """Constructor."""
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.use_residual = (stride == 1 and in_chs == out_chs)

        # Point-wise expansion
        self.conv_pw = nn.Conv2d(in_chs, mid_chs, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chs)
        self.act1 = nn.SiLU()

        # Depth-wise convolution
        padding = (dw_kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(mid_chs, mid_chs, kernel_size=dw_kernel_size, stride=stride, padding=padding, groups=mid_chs, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chs)
        self.act2 = nn.SiLU()

        # Squeeze-and-Excitation block (optional)
        self.se = SqueezeExcite(mid_chs, se_ratio) if use_se else nn.Identity()

        # Point-wise projection
        self.conv_pwl = nn.Conv2d(mid_chs, out_chs, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        """Forward pass."""
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.use_residual:
            x = x + shortcut
        return x


class SqueezeExcite(nn.Module):
    """Squeeze-and-excite. Component of EfficientNet B0."""
    def __init__(self, channels: int, se_ratio: float = .25):
        """"""
        super(SqueezeExcite, self).__init__()
        reduced_chs = max(1, int(round(channels * se_ratio)))
        self.conv_reduce = nn.Conv2d(channels, reduced_chs, kernel_size=1)
        self.act1 = nn.SiLU()
        self.conv_expand = nn.Conv2d(reduced_chs, channels, kernel_size=1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        """Forward pass."""
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return x * scale


def p_e_score(y_tr, y_pr):
    """Implementation of PE metric."""
    if (y_tr == 0).all() or (y_tr == 1).all():
        return np.nan
    # ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_tr, y_pr, pos_label=1, drop_intermediate=True)
    # NaNs in output
    if np.isnan(fpr).any() or np.isnan(tpr).any():
        return np.nan
    # P_E
    P = .5*(fpr + (1-tpr))
    return min(P[P > 0])
