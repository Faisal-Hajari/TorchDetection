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
    autopad,
)


def _validate_shortcut_compatibility(
    shortcut: bool, in_channels: int, out_channels: int
) -> None:
    """Validate that shortcut connection is compatible with channel dimensions."""
    if shortcut and in_channels != out_channels:
        raise ValueError(
            f"Shortcut requires in_channels == out_channels. "
            f"Got {in_channels} != {out_channels}. Set shortcut=False."
        )


class Concat(nn.Module):
    """Concatenate tensors along specified dimension."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.dim = dimension

    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=self.dim)


# Conv Blocks
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
        strict_depthwise: bool = True,
    ):
        """
        Initialize DepthwiseConv block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            dilation: Dilation
            strict_depthwise: if true, force use depthwise convolution.
            else it find the maximum number of groups that can be used.
            (groups = in_channels if strict_depthwise else math.gcd(in_channels, out_channels))

        Example:
            >>> dw_conv = DepthwiseConv(16, 16, 3, stride=1)
            >>> x = torch.randn(1, 16, 224, 224)
            >>> out = dw_conv(x)  # Shape: [1, 16, 224, 224]
        """
        super().__init__()
        if strict_depthwise and out_channels % in_channels != 0:
            raise ValueError(
                f"For depthwise conv, out_channels ({out_channels}) must be "
                f"divisible by in_channels ({in_channels})"
            )

        groups = (
            in_channels if strict_depthwise else math.gcd(in_channels, out_channels)
        )
        self.block = CBS(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=None,
            groups=groups,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DepthwiseConv block."""
        return self.block(x)


# Bottlenecks


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
        _validate_shortcut_compatibility(shortcut, in_channels, out_channels)

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

        _validate_shortcut_compatibility(shortcut, in_channels, out_channels)

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


class C3k(nn.Module):
    """CSP block with configurable kernel sizes. and 3 convolutions.

    A CSP (Cross Stage Partial) architecture block that uses configurable kernel
    bottlenecks. This block splits the input into two branches, processes one
    through multiple bottleneck layers, and concatenates the results.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        n_blocks: int = 1,
        kernel_size: tuple[int | tuple[int, int], int | tuple[int, int]] = (3, 3),
        shortcut: bool = True,
        groups: int = 1,
        bottleneck_expansion: float = 1.0,
    ):
        """Initialize C3k block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            expansion: Expansion factor for hidden channels
            n_blocks: Number of bottleneck blocks in the processing branch
            kernel_size: Kernel size for bottleneck convolutions
            shortcut: Whether to use shortcut connections in bottlenecks
            groups: Number of groups for the bottleneck

        Example:
            >>> block = C3k(64, 128, n_blocks=3, kernel_size=(5, 5))
            >>> x = torch.randn(1, 64, 224, 224)
            >>> out = block(x)  # Shape: [1, 128, 224, 224]
        """
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        self.cv1 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self.cv2 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self.cv3 = CBS(
            in_channels=(2 * hidden_channels),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.bottlenecks = nn.Sequential(
            *[
                Bottleneck(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_sizes=kernel_size,
                    shortcut=shortcut,
                    groups=groups,
                    expansion=bottleneck_expansion,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.cv2(x)
        branch2 = self.bottlenecks(self.cv1(x))
        return self.cv3(torch.cat((branch1, branch2), 1))


class C2F(nn.Module):
    """
    C2F (Faster CSP Bottleneck) block. with 2 convolutions.

    A CSP (Cross Stage Partial) architecture that splits input into two branches,
    processes one through multiple bottleneck layers, and concatenates all intermediate
    results.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int | tuple[int, int], int | tuple[int, int]] = (3, 3),
        shortcut: bool = False,
        n_blocks: int = 1,
        expansion: float = 0.5,
        groups: int = 1,
        bottleneck_expansion: float = 1.0,
    ):
        """
        Initialize C2F block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel sizes for bottleneck convolutions
            shortcut: Whether to use shortcut connections in bottlenecks
            n_blocks: Number of bottleneck blocks in the processing branch
            expansion: Expansion factor for CSP hidden channels
            groups: Number of groups for bottleneck convolutions
            bottleneck_expansion: Expansion factor within each bottleneck

        Example:
            >>> # Standard C2F block
            >>> block = C2F(64, 128, n_blocks=3)
            >>> x = torch.randn(1, 64, 224, 224)
            >>> out = block(x)  # Shape: [1, 128, 224, 224]
            >>>
            >>> # C2F with different kernel sizes
            >>> block = C2F(64, 128, kernel_size=((1, 1), (3, 3)))
        """
        super().__init__()

        self.hidden_channels = int(out_channels * expansion)

        self.cv1 = CBS(
            in_channels=in_channels,
            out_channels=(2 * self.hidden_channels),
            kernel_size=1,
            stride=1,
        )
        self.cv2 = CBS(
            in_channels=((2 + n_blocks) * self.hidden_channels),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.bottlenecks = nn.Sequential(
            *[
                Bottleneck(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_sizes=kernel_size,
                    shortcut=shortcut,
                    groups=groups,
                    expansion=bottleneck_expansion,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottlenecks)
        return self.cv2(torch.cat(y, 1))


class C3K2(nn.Module):
    """
    C3K2 (C3k CSP Bottleneck) block with 2 convolutions.

    A CSP (Cross Stage Partial) architecture that uses C3k bottlenecks instead of
    regular bottlenecks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int | tuple[int, int], int | tuple[int, int]] = (3, 3),
        shortcut: bool = True,
        n_blocks: int = 1,
        expansion: float = 0.5,
        groups: int = 1,
        bottleneck_expansion: float = 0.5,
        bottleneck_n_blocks: int = 2,
    ):
        """
        Initialize C3K2 block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel sizes for C3k bottleneck convolutions
            shortcut: Whether to use shortcut connections in C3k bottlenecks
            n_blocks: Number of C3k blocks in the processing branch
            expansion: Expansion factor for CSP hidden channels
            groups: Number of groups for C3k convolutions
            bottleneck_expansion: Expansion factor within each C3k block

        Example:
            >>> # Standard C3K2 block (YOLO11 style)
            >>> block = C3K2(64, 128, n_blocks=2)
            >>> x = torch.randn(1, 64, 224, 224)
            >>> out = block(x)  # Shape: [1, 128, 224, 224]
            >>>
            >>> # C3K2 with custom kernel sizes
            >>> block = C3K2(64, 128, kernel_size=((1, 1), (5, 5)), n_blocks=3)
        """
        super().__init__()

        self.hidden_channels = int(out_channels * expansion)

        self.cv1 = CBS(
            in_channels=in_channels,
            out_channels=(2 * self.hidden_channels),
            kernel_size=1,
            stride=1,
        )
        self.cv2 = CBS(
            in_channels=((2 + n_blocks) * self.hidden_channels),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.bottlenecks = nn.Sequential(
            *[
                C3k(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=kernel_size,
                    shortcut=shortcut,
                    groups=groups,
                    expansion=bottleneck_expansion,
                    n_blocks=bottleneck_n_blocks,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottlenecks)
        return self.cv2(torch.cat(y, 1))


# Pyramid blocks
class SPPF(nn.Module):
    """
    SPPF (Spatial Pyramid Pooling - Fast) block.

    A spatial pyramid pooling block that uses max pooling with different kernel sizes
    in sequence to capture multi-scale features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_kernels: list[int],
        expansion: float = 0.5,
    ) -> None:
        """
        Initialize SPPF block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            pool_kernels: List of kernel sizes for max pooling
            expansion: Expansion factor for hidden channels

        Example:
            >>> block = SPPF(64, 128, pool_kernels=[5, 9, 13])
            >>> x = torch.randn(1, 64, 224, 224)
            >>> out = block(x)  # Shape: [1, 128, 224, 224]
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self.pooling = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=kernel, stride=1, padding=autopad(kernel))
                for kernel in pool_kernels
            ]
        )
        self.cv2 = CBS(
            in_channels=hidden_channels * (len(pool_kernels) + 1),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(pooling(y[-1]) for pooling in self.pooling)
        return self.cv2(torch.cat(y, 1))


class SPP(nn.Module):
    """
    SPP (Spatial Pyramid Pooling) block.

    A spatial pyramid pooling block that uses max pooling with different kernel sizes
    in parallel to capture multi-scale features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_kernels: list[int],
        expansion: float = 0.5,
    ) -> None:
        """
        Initialize SPP block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            pool_kernels: List of kernel sizes for max pooling
            expansion: Expansion factor for hidden channels

        Example:
            >>> block = SPP(64, 128, pool_kernels=[5, 9, 13])
            >>> x = torch.randn(1, 64, 224, 224)
            >>> out = block(x)  # Shape: [1, 128, 224, 224]
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = CBS(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self.pooling = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=kernel, stride=1, padding=autopad(kernel))
                for kernel in pool_kernels
            ]
        )
        self.cv2 = CBS(
            in_channels=hidden_channels * (len(pool_kernels) + 1),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [pooling(x) for pooling in self.pooling], 1))


#HEADS 
class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module for bounding box regression.
    
    Converts probability distributions over discrete values into continuous coordinates
    through weighted averaging.
    """

    def __init__(self, reg_max: int = 16):
        """
        Initialize DFL module.
        
        Args:
            reg_max: Maximum regression range (number of discrete values)
        """
        super().__init__()
        self.reg_max = reg_max
        
        # Non-trainable 1x1 conv with fixed weights [0, 1, 2, ..., reg_max-1]
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        
        # Initialize weights as increasing sequence
        weight = torch.arange(reg_max, dtype=torch.float).view(1, reg_max, 1, 1)
        self.conv.weight.data = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DFL to convert distributions to coordinates.
        
        Args:
            x: Input tensor [B, 4*reg_max, H*W]
            
        Returns:
            Decoded coordinates [B, 4, H*W]
        """
        batch_size, _, num_anchors = x.shape
        
        # Reshape to [B, 4, reg_max, H*W] and apply softmax
        x = x.view(batch_size, 4, self.reg_max, num_anchors)
        x = x.transpose(2, 1).softmax(1)  # [B, reg_max, 4, H*W]
        
        # Apply weighted sum via convolution
        x = self.conv(x).view(batch_size, 4, num_anchors)
        
        return x
