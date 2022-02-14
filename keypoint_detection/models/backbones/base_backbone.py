import abc
import argparse

from torch import nn as nn


class Backbone(nn.Module, abc.ABC):
    def __init__(self):
        super(Backbone, self).__init__()

    @abc.abstractmethod
    def get_n_channels_out(self):
        pass

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parent_parser
