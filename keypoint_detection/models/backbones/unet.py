import argparse
import math

import torch
import torch.nn as nn

from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.backbones.s3k import ResNetBlock


class MaxPoolDownSamplingBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size):
        super().__init__()
        padding = math.floor(kernel_size / 2)
        self.conv = nn.Conv2d(
            in_channels=n_channels_in,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            stride=1,  # striding is a cheap way to downsample, but it is less informative that Pooling after full conv.
            dilation=1,  # dilation is a cheap way to increase receptive field, but it is less informative than deeper networks or downsampling..
            padding=padding,
            bias=False,  # with batchnorm, bias is ignored so optimize # params.
        )
        # extra sidenote stride + dilation -> equivalent to downsampling befor conv..
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self.norm = nn.BatchNorm2d(n_channels_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(
            x
        )  # activation and pool are commutative (if activation is monotonic), so pool first to reduce calculations
        x = self.norm(
            x
        )  # normalization can be used before or after activation: https://forums.fast.ai/t/order-of-layers-in-model/1261/3

        return x


class UpSamplingBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=n_channels_in * 2,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            bias=False,
            padding="same",
        )
        self.norm = nn.BatchNorm2d(n_channels_out)
        self.relu = nn.ReLU()

    def forward(self, x, x_skip):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)

        return x


class Unet(Backbone):
    def __init__(
        self, n_channels_in=3, n_downsampling_layers=2, n_resnet_blocks=3, n_channels=32, kernel_size=3, **kwargs
    ):
        super().__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size, padding="same")

        # create ModuleLists to ensure layers are discoverable by torch (lightning) for e.g. model summary and bringing to cuda.
        # https://pytorch.org/docs/master/generated/torch.nn.ModuleList.html#torch.nn.ModuleList
        self.downsampling_blocks = nn.ModuleList(
            [MaxPoolDownSamplingBlock(n_channels, n_channels, kernel_size) for _ in range(n_downsampling_layers)]
        )
        self.resnet_blocks = nn.ModuleList([ResNetBlock(n_channels, n_channels) for _ in range(n_resnet_blocks)])
        self.upsampling_blocks = nn.ModuleList(
            [
                UpSamplingBlock(n_channels_in=n_channels, n_channels_out=n_channels, kernel_size=kernel_size)
                for _ in range(n_downsampling_layers)
            ]
        )

    def forward(self, x):
        skips = []

        x = self.conv1(x)

        for block in self.downsampling_blocks:
            skips.append(x)
            x = block(x)

        for block in self.resnet_blocks:
            x = block(x)

        for block in self.upsampling_blocks:
            x_skip = skips.pop()
            x = block(x, x_skip)
        return x

    def get_n_channels_out(self):
        return self.n_channels

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("UnetBackbone")
        parser.add_argument("--n_channels_in", type=int, default=3)
        parser.add_argument("--n_channels", type=int, default=32)
        parser.add_argument("--n_resnet_blocks", type=int, default=3)
        parser.add_argument("--n_downsampling_layers", type=int, default=2)
        parser.add_argument("--kernel_size", type=int, default=3)

        return parent_parser
