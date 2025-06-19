"""Blocks for building detection models.

This module provides reusable blocks that can be composed to create more complex
architectures.
"""

import math

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
        kernel_size: int | tuple[int, int] = 1,
        stride: int = 1,
        padding: int | tuple[int, int] | None = None,
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
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        norm = nn.BatchNorm2d(out_channels)
        activation = nn.SiLU()

        self.block = PostActivationConvBlock(
            conv=conv, norm=norm, activation=activation
        )

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
        kernel_size: int | tuple[int, int] = 1,
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
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=None,
            groups=math.gcd(in_channels, out_channels),
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DepthwiseConv block."""
        return self.block(x)


class BaseBottleneck(nn.Module):
    """
    Base class for all bottleneck implementations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")


class Bottleneck(BaseBottleneck):
    """
    Bottleneck block with optional residual connection.

    Compresses the number of channels by the expansion factor, and optionally adds
    shortcut connection for gradient flow and feature preservation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        kernel_sizes: tuple[int | tuple[int, int], int | tuple[int, int]] = (3, 3),
        expansion: float = 0.5,
    ):
        """
        Initialize Bottleneck block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            shortcut: Add shortcut connection
            groups: Groups for the second convolution
            kernel_sizes: Kernel sizes for the two convolutions
            expansion: Expansion factor for the number of channels

        Example:
            >>> # Same dimensions - shortcut allowed
            >>> bottleneck = Bottleneck(64, 64, shortcut=True)
            >>> x = torch.randn(1, 64, 224, 224)
            >>> out = bottleneck(x)  # Shape: [1, 64, 224, 224]
            >>> # Different dimensions - must explicitly disable shortcut
            >>> bottleneck = Bottleneck(64, 128, shortcut=False)
        """
        super().__init__()
        if shortcut and in_channels != out_channels:
            raise ValueError(
                f"Shortcut requires in_channels == out_channels. "
                f"Got {in_channels} != {out_channels}. Set shortcut=False."
            )

        hidden_channels = int(out_channels * expansion)
        self.conv1 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
        )
        self.conv2 = CBS(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[1],
            stride=1,
            groups=groups,
        )
        self.add_shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional shortcut connection."""
        out = self.conv2(self.conv1(x))
        return x + out if self.add_shortcut else out


class DepthwiseBottleneck(BaseBottleneck):
    """
    Depthwise separable bottleneck for efficiency.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
    ):
        """
        Initialize DepthwiseBottleneck block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            shortcut: Add shortcut connection
            expansion: Expansion factor for the number of channels

        Example:
            >>> # Standard depthwise bottleneck
            >>> bottleneck = DepthwiseBottleneck(64, 64, shortcut=True)
            >>> x = torch.randn(1, 64, 224, 224)
            >>> out = bottleneck(x)  # Shape: [1, 64, 224, 224]
            >>> # Different dimensions - must explicitly disable shortcut
            >>> bottleneck = DepthwiseBottleneck(64, 128, shortcut=False)
        """
        super().__init__()

        if shortcut and in_channels != out_channels:
            raise ValueError(
                f"Shortcut requires in_channels == out_channels. "
                f"Got {in_channels} != {out_channels}. Set shortcut=False."
            )

        hidden_channels = int(out_channels * expansion)
        self.conv1 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
        )

        # Use depthwise convolution (groups = in_channels)
        self.conv2 = CBS(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            groups=hidden_channels,
        )
        self.add_shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.add_shortcut else out


