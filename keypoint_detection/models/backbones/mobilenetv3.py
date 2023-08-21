"""A MobileNetV3-based backbone.
"""
import timm
import torch
import torch.nn as nn

from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.backbones.convnext_unet import UpSamplingBlock


class MobileNetV3(Backbone):
    """
    Pretrained MobileNetV3 using the large_100 model with 3.4M parameters.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = timm.create_model("mobilenetv3_large_100", pretrained=True, features_only=True)
        self.decoder_blocks = nn.ModuleList()
        for i in range(1, len(self.encoder.feature_info.info)):
            channels_in, skip_channels_in = (
                self.encoder.feature_info.info[-i]["num_chs"],
                self.encoder.feature_info.info[-i - 1]["num_chs"],
            )
            block = UpSamplingBlock(channels_in, skip_channels_in, skip_channels_in, 3)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Conv2d(skip_channels_in, skip_channels_in, 3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        x = features.pop()
        for block in self.decoder_blocks:
            x = block(x, features.pop())
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.final_conv(x)

        return x

    def get_n_channels_out(self):
        return self.encoder.feature_info.info[0]["num_chs"]
