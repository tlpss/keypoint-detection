"""A backbone inspired by CoarseNet from the MinutiaeNet fingerprint detection model.
Adapted to work with the keypoint detection framework.
"""
import argparse
import torch
import torch.nn as nn

from keypoint_detection.models.backbones.base_backbone import Backbone


def conv_bn_prelu(in_channels, out_channels, kernel_size=3, stride=1, dilation_rate=1):
    """Convolutional block with BatchNorm and PReLU activation"""
    padding = kernel_size // 2 if dilation_rate == 1 else dilation_rate
    
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels, 
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation_rate,
        bias=False
    )
    
    bn = nn.BatchNorm2d(out_channels)
    prelu = nn.PReLU(num_parameters=out_channels, init=0.0)
    
    return nn.Sequential(conv, bn, prelu)


class ResidualBlock(nn.Module):
    """Residual block as used in CoarseNet"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_bn_prelu(in_channels, out_channels)
        self.conv2 = conv_bn_prelu(out_channels, out_channels)
        self.conv3 = conv_bn_prelu(out_channels, out_channels)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out += residual
        return out


class CoarseNetBackbone(Backbone):
    """
    Backbone inspired by the CoarseNet architecture from MinutiaeNet.
    Implements the main feature extraction components of CoarseNet.
    
    The architecture consists of:
    1. Initial convolution layers
    2. Series of residual blocks
    3. Multi-scale feature extraction
    """
    
    def __init__(self, n_channels=64, n_residual_blocks=3, **kwargs):
        super(CoarseNetBackbone, self).__init__()
        
        self.n_channels = n_channels
        input_channels = 3  # Standard RGB images
        
        # Initial layers
        self.conv1_0 = conv_bn_prelu(input_channels, n_channels, kernel_size=5)
        self.conv1_1 = conv_bn_prelu(n_channels, n_channels)
        self.conv1_2 = conv_bn_prelu(n_channels, n_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 1 with residual connections
        self.conv2_1 = conv_bn_prelu(n_channels, n_channels*2)
        self.conv2_2 = conv_bn_prelu(n_channels*2, n_channels*2)
        self.conv2_3 = conv_bn_prelu(n_channels*2, n_channels*2)
        
        # Additional residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(n_channels*2, n_channels*2) for _ in range(n_residual_blocks-1)
        ])
        
        # Downsampling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2 with more features
        self.conv3_1 = conv_bn_prelu(n_channels*2, n_channels*4)
        self.conv3_2 = conv_bn_prelu(n_channels*4, n_channels*4)
        self.conv3_3 = conv_bn_prelu(n_channels*4, n_channels*4)
        
        # Multiscale feature extraction
        self.level2 = conv_bn_prelu(n_channels*4, n_channels*4, dilation_rate=1)
        self.level3 = conv_bn_prelu(n_channels*4, n_channels*4, dilation_rate=4)
        self.level4 = conv_bn_prelu(n_channels*4, n_channels*4, dilation_rate=8)
        self.level5 = conv_bn_prelu(n_channels*4, n_channels*4, dilation_rate=16)
        
        # Final feature combination
        self.final_conv = nn.Conv2d(
            in_channels=n_channels*16,  # Concatenated features from 3 levels
            out_channels=n_channels*4,
            kernel_size=1,
            padding=0
        )
        self.final_bn = nn.BatchNorm2d(n_channels*4)
        self.final_relu = nn.ReLU(inplace=True)
        
        # Upsampling to restore resolution
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Initial processing
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        # First block
        conv1 = self.conv2_1(x)
        x = self.conv2_2(conv1)
        x = self.conv2_3(x)
        x = x + conv1
        
        # Additional residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Downsampling and second block
        x = self.pool2(x)
        
        conv2 = self.conv3_1(x)
        x = self.conv3_2(conv2)
        x = self.conv3_3(x)
        x = x + conv2
        
        # Multi-scale feature extraction
        feat2 = self.level2(x)
        feat3 = self.level3(x)
        feat4 = self.level4(x)
        feat5 = self.level5(x)
        
        # Concatenate features
        concat_feat = torch.cat([feat2, feat3, feat4, feat5], dim=1)
        
        # Final feature combination
        x = self.final_conv(concat_feat)
        x = self.final_bn(x)
        x = self.final_relu(x)
        
        # Upsample to restore resolution
        x = self.upsample(x)
        
        return x
    
    def get_n_channels_out(self):
        return self.n_channels * 4
    
    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("CoarseNetBackbone")
        parser.add_argument("--n_channels_coarse", dest="n_channels", type=int, default=64)
        parser.add_argument("--n_residual_blocks_coarse", dest="n_residual_blocks", type=int, default=3)
        return parent_parser
    
