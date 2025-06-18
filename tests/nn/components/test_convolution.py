import pytest
from torchdetection.nn.components.convolution import *
import torch
from torch import nn

class TestBaseClasses: 
    def test_base_conv(self):
        conv = BaseConv()
        with pytest.raises(NotImplementedError):
            conv(torch.randn(1, 3, 32, 32))
    
    def test_base_conv_block(self):
        block = BaseConvBlock()
        with pytest.raises(NotImplementedError):
            block(torch.randn(1, 3, 32, 32))

class TestConv2d:
    def test_forward(self):
        conv = Conv2d(3, 3, 7)
        input = torch.randn(1, 3, 32, 32)
        output = conv(input)
        assert output.shape == (1, 3, 32, 32)


class TestRepConv:
    def test_init_success(self):
        branches = [Conv2d(3, 16, 3), Conv2d(3, 16, 1)]
        repconv = RepConv(branches)
        assert isinstance(repconv.branches, nn.ModuleList)
        assert len(repconv.branches) == len(branches)
        assert repconv.reparametrized is False
        assert repconv.merged_conv is None

    def test_init_failure(self):
        # using nn.Conv2d
        branches = [nn.Conv2d(3, 16, 3), nn.Conv2d(3, 16, 1)]
        with pytest.raises(ValueError):
            RepConv(branches)

        # using different in_channels
        branches = [Conv2d(3, 16, 3), Conv2d(4, 16, 1)]
        with pytest.raises(ValueError):
            RepConv(branches)

        # using different out_channels
        branches = [Conv2d(3, 16, 3), Conv2d(3, 17, 1)]
        with pytest.raises(ValueError):
            RepConv(branches)

        # using different stride
        branches = [Conv2d(3, 16, 3, stride=1), Conv2d(3, 16, 3, stride=2)]
        with pytest.raises(ValueError):
            RepConv(branches)

        # using different groups
        branches = [Conv2d(16, 16, 3, groups=2), Conv2d(16, 16, 3, groups=1)]
        with pytest.raises(ValueError):
            RepConv(branches)

        # using different dilation
        branches = [Conv2d(3, 16, 3, dilation=1), Conv2d(3, 16, 3, dilation=2)]
        with pytest.raises(ValueError):
            RepConv(branches)

        # using even kernel_size
        branches = [Conv2d(3, 16, 4), Conv2d(3, 16, 1)]
        with pytest.raises(ValueError):
            RepConv(branches)

        # using non-square kernel_size
        branches = [Conv2d(3, 16, (3, 5)), Conv2d(3, 16, 1)]
        with pytest.raises(ValueError):
            RepConv(branches)
        
        #empty branches
        branches = []
        with pytest.raises(ValueError):
            RepConv(branches)

    def _evalute_reparametrize(self, branches: List[Conv2d], in_channels: int):
        repconv = RepConv(branches, atol=1e-5)
        input = torch.randn(1, in_channels, 32, 32)
        rep_conv_output = repconv(input)
        expected_output = sum([branch(input) for branch in branches])
        assert torch.allclose(rep_conv_output, expected_output)
        assert rep_conv_output.shape == expected_output.shape

    def test_forward_success(self):
        # basic kernel 3x3 and 1x1
        branches = [Conv2d(3, 16, 3), Conv2d(3, 16, 1)]
        self._evalute_reparametrize(branches, 3)

        # test with dialation
        branches = [Conv2d(3, 16, 3, dilation=2), Conv2d(3, 16, 1, dilation=2)]
        print(branches[0].groups)
        self._evalute_reparametrize(branches, 3)

        # test with groups
        branches = [Conv2d(4, 16, 3, groups=2), Conv2d(4, 16, 1, groups=2)]
        self._evalute_reparametrize(branches, 4)

        # test with many branches
        branches = [
            Conv2d(3, 16, 3),
            Conv2d(3, 16, 1),
            Conv2d(3, 16, 5),
            Conv2d(3, 16, 7),
        ]
        self._evalute_reparametrize(branches, 3)

    def test_reparametrize_success(self):
        branches = [
            Conv2d(3, 16, 3),
            Conv2d(3, 16, 1),
            Conv2d(3, 16, 5),
        ]
        repconv = RepConv(branches, atol=1e-5)
        input = torch.randn(1, 3, 32, 32)
        train_output = repconv(input)
        repconv.reparametrize()
        inference_output = repconv(input)

        print((train_output - inference_output).max())
        assert torch.allclose(train_output, inference_output, atol=1e-6)
        assert repconv.reparametrized is True
        assert repconv.merged_conv is not None

    def test_reparametrize_failure(self):
        branches = [Conv2d(3, 16, 3), Conv2d(3, 16, 1)]
        repconv = RepConv(branches, atol=1e-5)
        repconv.reparametrize()
        with pytest.raises(RuntimeError):
            repconv.reparametrize()
    

class TestPostActivationConvBlock:

    def test_forward(self):
        conv = Conv2d(3, 3, 7)
        norm = nn.BatchNorm2d(3)
        activation = nn.ReLU()
        block = PostActivationConvBlock(conv, norm, activation)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = activation(norm(conv(input)))
        assert output.shape == (1, 3, 32, 32)
        assert torch.allclose(output, expected_output)

    def _test_excpected_output(self,  conv:Conv2d, norm:nn.Module, activation:nn.Module):
        block = PostActivationConvBlock(conv, norm, activation)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = activation(norm(conv(input)))
        assert output.shape == (1, 3, 32, 32)
        assert torch.allclose(output, expected_output)

    def test_with_different_norms(self):
        #batch norm 
        self._test_excpected_output(Conv2d(3, 3, 7), nn.BatchNorm2d(3), nn.ReLU())

        #group norm 
        self._test_excpected_output(Conv2d(3, 3, 7), nn.GroupNorm(3, 3), nn.ReLU())

        #instance norm 
        self._test_excpected_output(Conv2d(3, 3, 7), nn.InstanceNorm2d(3), nn.ReLU())

        #layer norm 
        self._test_excpected_output(Conv2d(3, 3, 7), nn.LayerNorm([3, 32, 32]), nn.ReLU())

        #rms norm 
        self._test_excpected_output(Conv2d(3, 3, 7), nn.RMSNorm([3, 32, 32]), nn.ReLU())

        #sync batch norm 
        self._test_excpected_output(Conv2d(3, 3, 7), nn.SyncBatchNorm(3), nn.ReLU())

        #local response norm 
        self._test_excpected_output(Conv2d(3, 3, 7), nn.LocalResponseNorm(3), nn.ReLU())

    def test_edge_cases(self):
        """Test important edge cases for PostActivation"""
        
        # No normalization (norm=None)
        conv = Conv2d(3, 16, 7)
        activation = nn.ReLU()
        block = PostActivationConvBlock(conv, norm=None, activation=activation)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = activation(conv(input))  # Just conv → activation
        assert output.shape == (1, 16, 32, 32)
        assert torch.allclose(output, expected_output)

        # No activation (activation=None)
        conv = Conv2d(3, 16, 7)
        norm = nn.BatchNorm2d(16)  # NOTE: norm on OUTPUT channels for PostActivation
        block = PostActivationConvBlock(conv, norm=norm, activation=None)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = norm(conv(input))  # Just conv → norm
        assert output.shape == (1, 16, 32, 32)
        assert torch.allclose(output, expected_output)

        # Neither norm nor activation (just conv)
        conv = Conv2d(3, 16, 7)
        block = PostActivationConvBlock(conv, norm=None, activation=None)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = conv(input)  # Just conv
        assert output.shape == (1, 16, 32, 32)
        assert torch.allclose(output, expected_output)

    def test_with_different_activations(self):
        """Test various activation functions"""
        conv = Conv2d(3, 16, 7)
        norm = nn.BatchNorm2d(16)  # On output channels for PostActivation
        
        activations = [
            nn.ReLU(),
            nn.LeakyReLU(),
            nn.ELU(),
            nn.SiLU(),      # Swish - important for YOLOv11!
            nn.GELU(),
            nn.Mish(),
            nn.Hardswish(),
            nn.ReLU6(),
            nn.Tanh(),
            nn.Sigmoid(),
        ]
        
        input = torch.randn(1, 3, 32, 32)
        
        for activation in activations:
            block = PostActivationConvBlock(conv, norm, activation)
            output = block(input)
            expected_output = activation(norm(conv(input)))  # conv → norm → activation
            assert output.shape == (1, 16, 32, 32)
            assert torch.allclose(output, expected_output)

    def test_silu_activation_for_yolo(self):
        """SiLU/Swish activation - CRITICAL for YOLOv11 implementation!"""
        conv = Conv2d(3, 16, 7)
        norm = nn.BatchNorm2d(16)  # On output channels
        activation = nn.SiLU()  # This will be used in YOLOv11 CBS blocks
        block = PostActivationConvBlock(conv, norm, activation)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = activation(norm(conv(input)))  # conv → norm → activation
        assert output.shape == (1, 16, 32, 32)
        assert torch.allclose(output, expected_output)

class TestPreActivationConvBlock:

    def test_forward(self):
        conv = Conv2d(3, 16, 7)  # 3 input channels, 16 output channels
        norm = nn.BatchNorm2d(3)  # NOTE: norm on INPUT channels (3), not output (16)
        activation = nn.ReLU()
        block = PreActivationConvBlock(conv, norm, activation)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        # PreActivation order: norm → activation → conv
        expected_output = conv(activation(norm(input)))
        assert output.shape == (1, 16, 32, 32)  # Output has 16 channels
        assert torch.allclose(output, expected_output)

    def _test_expected_output(self, conv: Conv2d, norm: nn.Module, activation: nn.Module):
        block = PreActivationConvBlock(conv, norm, activation)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        # PreActivation order: norm → activation → conv
        expected_output = conv(activation(norm(input)))
        assert output.shape == (1, conv.out_channels, 32, 32)
        assert torch.allclose(output, expected_output)

    def test_with_different_norms(self):
        # NOTE: All norms are on INPUT channels (3), not output channels!
        
        # batch norm 
        self._test_expected_output(Conv2d(3, 16, 7), nn.BatchNorm2d(3), nn.ReLU())

        # group norm 
        self._test_expected_output(Conv2d(3, 16, 7), nn.GroupNorm(3, 3), nn.ReLU())

        # instance norm 
        self._test_expected_output(Conv2d(3, 16, 7), nn.InstanceNorm2d(3), nn.ReLU())

        # layer norm 
        self._test_expected_output(Conv2d(3, 16, 7), nn.LayerNorm([3, 32, 32]), nn.ReLU())

        # rms norm 
        self._test_expected_output(Conv2d(3, 16, 7), nn.RMSNorm([3, 32, 32]), nn.ReLU())

        # sync batch norm 
        self._test_expected_output(Conv2d(3, 16, 7), nn.SyncBatchNorm(3), nn.ReLU())

        # local response norm 
        self._test_expected_output(Conv2d(3, 16, 7), nn.LocalResponseNorm(3), nn.ReLU())

    def test_edge_cases(self):
        """Test important edge cases for PreActivation"""
        
        # No normalization (norm=None)
        conv = Conv2d(3, 16, 7)
        activation = nn.ReLU()
        block = PreActivationConvBlock(conv, norm=None, activation=activation)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = conv(activation(input))  # Just activation → conv
        assert output.shape == (1, 16, 32, 32)
        assert torch.allclose(output, expected_output)

        # No activation (activation=None)
        conv = Conv2d(3, 16, 7)
        norm = nn.BatchNorm2d(3)
        block = PreActivationConvBlock(conv, norm=norm, activation=None)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = conv(norm(input))  # Just norm → conv
        assert output.shape == (1, 16, 32, 32)
        assert torch.allclose(output, expected_output)

        # Neither norm nor activation (just conv)
        conv = Conv2d(3, 16, 7)
        block = PreActivationConvBlock(conv, norm=None, activation=None)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = conv(input)  # Just conv
        assert output.shape == (1, 16, 32, 32)
        assert torch.allclose(output, expected_output)

    def test_with_different_activations(self):
        """Test various activation functions"""
        conv = Conv2d(3, 16, 7)
        norm = nn.BatchNorm2d(3)  # On input channels
        
        activations = [
            nn.ReLU(),
            nn.LeakyReLU(),
            nn.ELU(),
            nn.SiLU(),      # Swish - important for YOLOv11!
            nn.GELU(),
            nn.Mish(),
            nn.Hardswish(),
            nn.ReLU6(),
            nn.Tanh(),
            nn.Sigmoid(),
        ]
        
        input = torch.randn(1, 3, 32, 32)
        
        for activation in activations:
            block = PreActivationConvBlock(conv, norm, activation)
            output = block(input)
            expected_output = conv(activation(norm(input)))
            assert output.shape == (1, 16, 32, 32)
            assert torch.allclose(output, expected_output)

    def test_silu_activation_for_yolo(self):
        """SiLU/Swish activation - CRITICAL for YOLOv11 implementation!"""
        conv = Conv2d(3, 16, 7)
        norm = nn.BatchNorm2d(3)
        activation = nn.SiLU()  # This will be used in YOLOv11
        block = PreActivationConvBlock(conv, norm, activation)
        input = torch.randn(1, 3, 32, 32)
        output = block(input)
        expected_output = conv(activation(norm(input)))
        assert output.shape == (1, 16, 32, 32)
        assert torch.allclose(output, expected_output)