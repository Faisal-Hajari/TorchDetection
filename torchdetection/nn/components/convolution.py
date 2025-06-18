"""Convolutional components for building detection models.

This module provides reusable convolutional building blocks that can be composed
to create more complex architectures. Each component follows PyTorch conventions
and maintains explicit dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple


def autopad(
    kernel_size: Union[int, List[int]],
    padding: Optional[Union[int, List[int]]] = None,
    dilation: int = 1,
) -> Union[int, List[int]]:
    """Returns the padding for a given kernel size, padding, and dilation.
    that maintains the same output shape as the input shape.

    Args:
        kernel_size: int or list of ints
        padding: int or list of ints
        dilation: int

    Returns:
        padding: int or list of ints
    """
    if dilation > 1:
        kernel_size = (
            dilation * (kernel_size - 1) + 1
            if isinstance(kernel_size, int)
            else [dilation * (x - 1) + 1 for x in kernel_size]
        )
    if padding is None:
        padding = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else [x // 2 for x in kernel_size]
        )
    return padding


class BaseConv(nn.Module):
    r"""
    Base class for all convolutional layers.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")


class Conv2d(BaseConv):
    """Standard conv2d layer

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size.
        stride (int): Stride.
        padding (int, optional): Padding. If None, padding is calculated to maintain output shape.
        groups (int): Groups.
        dilation (int): Dilation.
        bias (bool): Whether to use bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(
                autopad(kernel_size, padding, dilation) if padding is None else padding
            ),
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.padding = (
            autopad(kernel_size, padding, dilation) if padding is None else padding
        )
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.conv.bias


class RepConv(BaseConv):
    """Flexible Reparameterized Convolution block that can merge multiple branches during inference.
    having multiple branches during training, and merge them during inference. helps with gradient
    flow and training performance, while maintaining the same inference time as a single branch.
    the layer can be used as a drop-in replacement for Conv2d. andit should be reparametrized
    before exporting the model.

    Notes:
    - currently only support odd kernel sizes.
    - the branches should maintain the shape of the input (H_in, W_in) == (H_out, W_out).
    - there is a difference between the output of the branches and the merged conv. with a
      tolerance of 1e-6.

    reference: https://arxiv.org/abs/2101.03697
    """

    def __init__(
        self,
        branches: List[Conv2d],
        atol: float = 1e-6,
    ) -> None:
        """Initialize the RepConv layer.

        Args:
            branches: List of nn.Conv2d branches.
            atol: float, the tolerance for the difference between the output of the branches and
            the merged conv.
        """
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.reparametrized = False
        self.merged_conv: Union[Conv2d, None] = None
        self.atol = atol
        self._validate_branches(self.branches, self.atol)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            output: SUM(branches(x)). Returns the sum of the output of the branches.
        """
        if self.reparametrized:
            return self.merged_conv(x)

        output = sum([branch(x) for branch in self.branches])
        return output

    def reparametrize(self) -> None:
        """Reparametrize the RepConv layer by merging Conv2d branches.

        This method fuses the parameters of all Conv2d branches into a single
        convolutional layer, which is more efficient for inference.
        After reparametrization, the original branches are removed.
        """
        if self.reparametrized:
            raise RuntimeError("RepConv is already reparametrized")

        self.merged_conv = self._create_merged_conv(self.branches)
        self.reparametrized = True

        # Free up memory
        del self.branches

    @torch.no_grad()
    def _validate_branches(self, branches: nn.ModuleList, atol: float) -> None:
        """Validate that all branches are compatible for merging."""
        if len(branches) < 1:
            raise ValueError("RepConv requires at least one branch")

        for branch in branches:
            if not isinstance(branch, Conv2d):
                raise ValueError("All branches must be Conv2d")
            if branch.in_channels != branches[0].in_channels:
                raise ValueError("All branches must have the same in_channels")
            if branch.out_channels != branches[0].out_channels:
                raise ValueError("All branches must have the same out_channels")
            if branch.stride != branches[0].stride:
                raise ValueError("All branches must have the same stride")
            if branch.groups != branches[0].groups:
                raise ValueError("All branches must have the same groups")
            if branch.dilation != branches[0].dilation:
                raise ValueError("All branches must have the same dilation")
            if branch.kernel_size[0] != branch.kernel_size[1]:
                raise ValueError("All branches must have square kernel_size")
            if branch.kernel_size[0] % 2 == 0:
                raise ValueError("All branches must have odd kernel_size")

        # is this good ? (we are hard evaluating if the settings are correct)
        merged_conv = self._create_merged_conv(self.branches)
        max_k = max(conv.kernel_size[0] for conv in branches)
        input = torch.randn(1, merged_conv.in_channels, max_k * 2, max_k * 2)
        merged_conv_output = merged_conv(input)
        branches_output = sum([branch(input) for branch in branches])
        error = (merged_conv_output - branches_output).abs().max()
        assert (
            error < atol
        ), f"Merged conv does not produce the same output as the branches. the error is {error}"

    @torch.no_grad()
    def _create_merged_conv(self, branches: nn.ModuleList) -> nn.Conv2d:
        """Creates a single Conv2d layer by merging all branches."""
        largest_branch = max(branches, key=lambda x: x.kernel_size[0])
        kernel_size = largest_branch.kernel_size
        stride = largest_branch.stride
        padding = largest_branch.padding
        groups = largest_branch.groups
        dilation = largest_branch.dilation
        in_channels = largest_branch.in_channels
        out_channels = largest_branch.out_channels
        device = largest_branch.weight.device

        merged_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups
        ).to(device)
        merged_conv.weight.data *= 0
        merged_conv.bias.data *= 0

        for branch in branches:
            padding = (kernel_size[0] - branch.kernel_size[0]) // 2
            padded_wieghts = F.pad(
                branch.weight, [padding] * 4, mode="constant", value=0
            )
            merged_conv.weight.data += padded_wieghts
            merged_conv.bias.data += branch.bias.data

        return merged_conv


class BaseConvBlock(nn.Module):
    """Base class for all convolutional blocks."""

    def __init__(self):
        super().__init__()


class PostActivationConvBlock(BaseConvBlock):
    r"""
    Standard convolutional block with Conv2d + normalization + activation.
    """

    def __init__(
        self,
        conv: nn.Conv2d | BaseConv,
        norm: nn.Module | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        """Initialize the convolutional block.

        Args:
            conv: Convolutional layer (nn.Conv2d or BaseConv)
            norm: Optional normalization layer (e.g. BatchNorm2d). Default: None
            activation: Optional activation function (e.g. ReLU). Default: None

        Order: input → conv → norm → activation → output

        Example:
            >>> conv = Conv(nn.Conv2d(3, 16, 3, padding=1), norm=nn.BatchNorm2d(16),
            ...             activation=nn.ReLU())
            >>> x = torch.randn(1, 3, 224, 224)
            >>> out = conv(x)  # Shape: [1, 16, 224, 224]
        """
        super().__init__()
        self.conv = conv
        self.norm = norm
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Output tensor of shape [bat
            x: Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            Output tensor of shape [batch_size, out_channels, height', width']
            where height' and width' depend on kernel_size, stride, and padding
        """
        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class PreActivationConvBlock(BaseConvBlock):
    r"""
    Pre-activation convolutional block with normalization + activation + Conv2d.

    Applies normalization and activation before convolution, which can improve
    gradient flow and training stability. Used for deep networks like ResNet1001.

    reference: https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(
        self,
        conv: nn.Conv2d | BaseConv,
        norm: nn.Module | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        """Initialize the pre-activation convolutional block.

        Args:
            conv: Convolutional layer (nn.Conv2d or BaseConv)
            norm: Optional normalization layer (e.g. BatchNorm2d). Default: None
            activation: Optional activation function (e.g. ReLU). Default: None

        Order: input → norm → activation → conv → output

        Example:
            >>> conv = PreActivationConv(nn.Conv2d(3, 16, 3, padding=1),
            ...                         norm=nn.BatchNorm2d(3), activation=nn.ReLU())
            >>> x = torch.randn(1, 3, 224, 224)
            >>> out = conv(x)  # Shape: [1, 16, 224, 224]
        """
        super().__init__()
        self.norm = norm
        self.activation = activation
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            Output tensor of shape [batch_size, out_channels, height', width']
            where height' and width' depend on kernel_size, stride, and padding
        """
        # Pre-activation: norm and activation come before convolution
        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        x = self.conv(x)
        return x