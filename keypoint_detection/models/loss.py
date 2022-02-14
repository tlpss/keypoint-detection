import argparse

import torch
import torch.nn.functional as functional


def focal_loss():
    pass


def bce_loss(predicted_heatmaps: torch.Tensor, heatmaps: torch.Tensor) -> torch.Tensor:
    return functional.binary_cross_entropy(predicted_heatmaps, heatmaps, reduction="mean")


class LossFactory:
    @staticmethod
    def create_loss(loss: str, **kwargs):
        if loss == "bce":
            return bce_loss
        else:
            raise Exception("Unknown loss type")

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("LossFactory")
        parser.add_argument("--loss", type=str, default="bce")
        return parent_parser
