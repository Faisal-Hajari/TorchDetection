"""Convolutional components for building detection models.

This module provides reusable convolutional building blocks that can be composed
to create more complex architectures. Each component follows PyTorch conventions
and maintains explicit dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple       

class BaseConv(nn.Module):
    r"""
    Base class for all convolutional layers.
    """
    def __init__(self): 
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")
       
        
class RepConv(BaseConv):
    """Flexible Reparameterized Convolution block that can merge multiple branches during inference.
    having multiple branches during training, and merge them during inference. helps with gradient flow and training performance,
    while maintaining the same inference time as a single branch. the layer can be used as a drop-in replacement for Conv2d. and 
    it should be reparametrized before exporting the model.
    
    reference: https://arxiv.org/abs/2101.03697
    """
    def __init__(
        self,
        branches: List[nn.Conv2d],
    ) -> None:
        """Initialize the RepConv layer.
        
        Args:
            branches: List of nn.Conv2d branches.
        """
        super().__init__()
        assert len(branches) > 0, "RepConv requires at least one branch"
        self.branches = nn.ModuleList(branches)
        self.reparametrized = False
        self.merged_conv: Optional[nn.Conv2d] = None
        
        self._validate_branches()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            output: SUM(branches(x)). Returns the sum of the output of the branches.
        """
        if self.reparametrized:
            return self.merged_conv(x)
        
        output = sum(branch(x) for branch in self.branches)
        return output
    
    def reparametrize(self) -> None:
        """Reparametrize the RepConv layer by merging Conv2d branches.
        
        This method fuses the parameters of all Conv2d branches into a single
        convolutional layer, which is more efficient for inference.
        After reparametrization, the original branches are removed.
        """
        if self.reparametrized:
            raise RuntimeError("RepConv is already reparametrized")
            
        self.merged_conv = self._create_merged_conv()
        self.reparametrized = True
        
        # Free up memory
        del self.branches

    def _validate_branches(self) -> None:
        """Validate that all branches are compatible for merging."""
        first_conv = self.branches[0]
        for conv in self.branches[1:]:
            if not isinstance(conv, nn.Conv2d):
                raise TypeError(f"All branches must be nn.Conv2d, but found {type(conv)}")
            if conv.in_channels != first_conv.in_channels:
                raise ValueError("All branches must have the same in_channels.")
            if conv.out_channels != first_conv.out_channels:
                raise ValueError("All branches must have the same out_channels.")
            if conv.stride != first_conv.stride:
                raise ValueError("All branches must have the same stride.")
            if conv.groups != first_conv.groups:
                raise ValueError("All branches must have the same groups.")

    def _create_merged_conv(self) -> nn.Conv2d:
        """Creates a single Conv2d layer by merging all branches."""
        first_conv = self.branches[0]
        in_channels = first_conv.in_channels
        out_channels = first_conv.out_channels
        stride = first_conv.stride
        groups = first_conv.groups
        device = first_conv.weight.device

        # Find max kernel size for padding
        max_k = max(conv.kernel_size[0] for conv in self.branches)
        padding = max_k // 2

        # Sum kernels and biases
        total_kernel = torch.zeros(out_channels, in_channels // groups, max_k, max_k, device=device)
        total_bias = torch.zeros(out_channels, device=device)

        for conv in self.branches:
            kernel = conv.weight
            bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=device)
            
            pad = (max_k - kernel.shape[2]) // 2
            padded_k = F.pad(kernel, [pad, pad, pad, pad])
            
            total_kernel += padded_k
            total_bias += bias

        # Create and set merged conv
        merged_conv = nn.Conv2d(in_channels, out_channels, max_k, stride, padding, groups=groups, bias=True)
        merged_conv.to(device)
        merged_conv.weight.data = total_kernel
        merged_conv.bias.data = total_bias
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
        activation: nn.Module | None = None
    )-> None:
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
        activation: nn.Module | None = None
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
