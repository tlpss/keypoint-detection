import unittest

import torch

from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.backbones.maxvit_unet import MaxVitUnet


class TestBackbones(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_output_format_of_all_registered_backbones(self):
        kwargs = {
            "n_channels": 32,
            "kernel_size": 5,
        }
        for backbone in BackboneFactory.registered_backbone_classes:
            model: Backbone = backbone(**kwargs).to(self.device)
            shape = (4, 3, 64, 64)
            if isinstance(model, MaxVitUnet):
                # MaxVit can deal with multiples of 32 as input shape,
                #  but pretraind weights are only available for 224,256,...
                # and for now only 256 is implemented (see MaxViTUnet class)
                shape = (4, 3, 256, 256)

            x = torch.randn(shape).to(self.device)
            output = model(x).cpu()
            self.assertEqual(output.shape[0], shape[0])
            self.assertEqual(output.shape[-2:], shape[-2:])
            self.assertEqual(output.shape[1], model.get_n_channels_out())
