import pytest
from torchdetection.nn.components.convolution import *
import torch
from torch import nn


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
