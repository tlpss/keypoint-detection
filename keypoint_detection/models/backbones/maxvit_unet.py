import timm
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.backbones.convnext_unet import UpSamplingBlock


class MaxVitUnet(Backbone):
    """
    Pretrained MaxVit(MBConv (Efficient Net) + Blocked Local Attention + Grid global attention) as Encoder for the U-Net.

    the outputs of the stem and all Multi-Axis (Max) stages are used as feature layers
    note that the paper uses only stage 2-4 for segmentation w/ Mask-RCNN.

    maxvit_nano_rw_256 is a version trained on 256x256 images in timm, that differs slightly from the paper
    but is a much more lightweight model (approx. 15M params)

    It is approx 4 times slower than the ConvNeXt femto backbone (5M params), and still
    about 2 times slower than convnext_nano @ 15M params, yet provided better results
    than both convnext variants in some initial experiments.

    The model can deal with input sizes divisible by 32, but for pretrained weights you are restricted to multiples of the pretrained
    models: 224, 256, 384. From the accompanying notebook, it seems that the model easily handles images that are 3 times as big as the
    training size. (see https://github.com/rwightman/pytorch-image-models/issues/1475 for more details)

    For now only 256 is supported so input sizes are restricted to 256,512,...


                                                        (head)
    stem                ---   1/2  -->             final_up (bilinear 2x)
        stage 1         ---   1/4  -->         decode3
            stage 2     ---   1/8  -->     decode2
                stage 3 ---   1/16 --> decode1
                    stage 4 ---1/32----|
    """

    # manually gathered for maxvit_nano_rw_256
    feature_config = [
        {"down": 2, "channels": 64},
        {"down": 4, "channels": 64},
        {"down": 8, "channels": 128},
        {"down": 16, "channels": 256},
        {"down": 32, "channels": 512},
    ]
    feature_layers = ["stem", "stages.0", "stages.1", "stages.2", "stages.3"]

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder = timm.create_model("maxvit_nano_rw_256", pretrained=True, num_classes=0)  # 15M params
        self.feature_extractor = create_feature_extractor(self.encoder, self.feature_layers)
        self.decoder_blocks = nn.ModuleList()
        for config_skip, config_in in zip(self.feature_config, self.feature_config[1:]):
            block = UpSamplingBlock(config_in["channels"], config_skip["channels"], config_skip["channels"], 3)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Conv2d(
            self.feature_config[0]["channels"], self.feature_config[0]["channels"], 3, padding="same"
        )

    def forward(self, x):
        features = list(self.feature_extractor(x).values())
        x = features.pop(-1)
        for block in self.decoder_blocks[::-1]:
            x = block(x, features.pop(-1))
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.final_conv(x)
        return x

    def get_n_channels_out(self):
        return self.feature_config[0]["channels"]


if __name__ == "__main__":
    # model = timm.create_model("maxvit_rmlp_pico_rw_256")
    model = timm.create_model("maxvit_nano_rw_256")
    feature_extractor = create_feature_extractor(model, ["stem", "stages.0", "stages.1", "stages.2", "stages.3"])
    x = torch.zeros((1, 3, 256, 256))
    features = list(feature_extractor(x).values())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num params = {n_params/10**6:.2f} M")
    feature_config = []
    for x in features:
        print(f"{x.shape=}")
        config = {"down": 256 // x.shape[2], "channels": x.shape[1]}
        feature_config.append(config)
    print(f"{feature_config=}")
