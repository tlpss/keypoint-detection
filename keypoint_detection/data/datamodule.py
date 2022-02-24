import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class RandomSplitDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        """
        parser = parent_parser.add_argument_group("RandomSplitDatamodule")
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--validation_split_ratio", default=0.25, type=float)
        parser.add_argument("--num_workers", default=4, type=int)

        return parent_parser

    def __init__(self, dataset: Dataset, batch_size, validation_split_ratio, num_workers, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        validation_size = int(validation_split_ratio * len(self.dataset))
        train_size = len(self.dataset) - validation_size
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, validation_size]
        )

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return dataloader

    def val_dataloader(self):
        # num workers to zero to avoid non-reproducibility bc of random seeds for workers
        # cf. https://pytorch.org/docs/stable/notes/randomness.html
        dataloader = DataLoader(self.validation_dataset, self.batch_size, shuffle=False, num_workers=0)
        return dataloader

    def test_dataloader(self):
        pass
