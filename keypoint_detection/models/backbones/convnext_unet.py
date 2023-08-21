"""A Unet-like backbone that uses a (relatively) small imagenet-pretrained ConvNeXt model from timm as encoder.
"""
import timm
import torch
import torch.nn as nn

from keypoint_detection.models.backbones.base_backbone import Backbone


class UpSamplingBlock(nn.Module):
    """
    A very basic Upsampling block (these params have to be learnt from scratch so keep them small)

    First it reduces the number of channels of the incoming layer to the amount of the skip connection with a 1x1 conv
    then it concatenates them and combines them in a new conv layer.



    x --> up ---> conv1 --> concat --> conv2 --> norm -> relu
                  ^
                  |
                  skip_x
    """

    def __init__(self, n_channels_in, n_skip_channels_in, n_channels_out, kernel_size):
        super().__init__()
        # bilinear is not deterministic, use nearest neighbor instead
        self.upsample = lambda x: nn.functional.interpolate(x, scale_factor=2)
        self.conv_reduce = nn.Conv2d(
            in_channels=n_channels_in, out_channels=n_skip_channels_in, kernel_size=1, bias=False, padding="same"
        )
        self.conv = nn.Conv2d(
            in_channels=n_skip_channels_in * 2,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            bias=False,
            padding="same",
        )
        self.norm = nn.BatchNorm2d(n_channels_out)
        self.relu = nn.ReLU()

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = self.conv_reduce(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class ConvNeXtUnet(Backbone):
    """
    Pretrained ConvNeXt as Encoder for the U-Net.

    the outputs of the 3 intermediate CovNext stages are used for skip connections.
    The output of res4 is considered as the bottleneck and has a 32x resolution reduction!

    femto -> 3M params
    nano -> 17M params (but only twice as slow)


                                        (head)
    stem                              final_up ( 4x)
        res1         --->   1/4      decode3
            res2     --->   1/8    decode2
                res3 --->   1/16  decode1
                    res4 ---1/32----|
    """

    def __init__(self, **kwargs):
        super().__init__()
        # todo: make desired convnext encoder configurable
        self.encoder = timm.create_model("convnext_femto", features_only=True, pretrained=True)

        self.decoder_blocks = nn.ModuleList()
        for i in range(1, 4):
            channels_in, skip_channels_in = (
                self.encoder.feature_info.info[-i]["num_chs"],
                self.encoder.feature_info.info[-i - 1]["num_chs"],
            )
            block = UpSamplingBlock(channels_in, skip_channels_in, skip_channels_in, 3)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Conv2d(skip_channels_in, skip_channels_in, 3, padding="same")

    def forward(self, x):
        features = self.encoder(x)

        x = features.pop()
        for block in self.decoder_blocks:
            x = block(x, features.pop())
        x = nn.functional.interpolate(x, scale_factor=4)
        x = self.final_conv(x)
        return x

    def get_n_channels_out(self):
        return self.encoder.feature_info.info[0]["num_chs"]
