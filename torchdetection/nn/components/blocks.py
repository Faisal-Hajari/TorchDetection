"""Blocks for building detection models.

This module provides reusable blocks that can be composed to create more complex
architectures.
"""

import math
from typing import Optional, Union, Tuple

import torch
from torch import nn

from torchdetection.nn.components.convolution import (
    Conv2d,
    PostActivationConvBlock,
)


class CBS(nn.Module):
    """
    Conv2d(bias=False) → BatchNorm2d → SiLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: int = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        groups: int = 1,
        dilation: int = 1,
    ):
        """
        Initialize CBS block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding (auto-calculated if None)
            groups: Groups (1=normal conv, in_channels=depthwise conv)
            dilation: Dilation

        Example:
            >>> # Standard CBS block
            >>> cbs = CBS(3, 16, 3)
            >>> # Depthwise CBS block
            >>> dw_cbs = CBS(16, 16, 3, groups=16)
            >>> # Stride 2 for downsampling
            >>> down_cbs = CBS(16, 32, 3, stride=2)
        """
        super().__init__()

        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation,
            bias=False,
        )
        norm = nn.BatchNorm2d(out_channels)
        activation = nn.SiLU()

        self.block = PostActivationConvBlock(conv, norm, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CBS block."""
        return self.block(x)


class DepthwiseConv(nn.Module):
    """
    Depthwise Convolutional Block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: int = 1,
        dilation: int = 1,
    ):
        """
        Initialize DepthwiseConv block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            dilation: Dilation

        Example:
            >>> dw_conv = DepthwiseConv(16, 16, 3, stride=1)
            >>> x = torch.randn(1, 16, 224, 224)
            >>> out = dw_conv(x)  # Shape: [1, 16, 224, 224]
        """
        super().__init__()
        self.block = CBS(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            None,
            math.gcd(in_channels, out_channels),
            dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DepthwiseConv block."""
        return self.block(x)
