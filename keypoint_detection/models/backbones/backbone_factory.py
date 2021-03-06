import argparse

from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.backbones.dilated_cnn import DilatedCnn
from keypoint_detection.models.backbones.s3k import S3K
from keypoint_detection.models.backbones.unet import UnetBackbone


class BackboneFactory:
    @staticmethod
    def create_backbone(backbone_type: str, **kwargs) -> Backbone:
        if backbone_type == "DilatedCnn":
            return DilatedCnn(**kwargs)
        elif backbone_type == "S3K":
            return S3K()
        elif backbone_type == "Unet":
            return UnetBackbone(**kwargs)
        else:
            raise Exception("Unknown backbone type")

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("BackboneFactory")
        parser.add_argument(
            "--backbone_type", type=str, default="Unet", help="The Class of the Backbone for the Detector."
        )

        # add all possible backbone hyperparams.
        # would be better to check at runtime if no arguments of other backbones have been added as these will be ignored
        parent_parser = DilatedCnn.add_to_argparse(parent_parser)
        parent_parser = S3K.add_to_argparse(parent_parser)
        parent_parser = UnetBackbone.add_to_argparse(parent_parser)

        return parent_parser
