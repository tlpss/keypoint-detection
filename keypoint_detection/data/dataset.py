import abc
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import tqdm
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from keypoint_detection.utils.tensor_padding import pad_tensor_with_nans


class ImageDataset(Dataset, abc.ABC):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def get_image(self, index: int) -> np.ndarray:
        """
        get image associated to dataset[index]
        """


class KeypointsDataset(ImageDataset):
    """
    Create Custom Pytorch Dataset from the Box dataset
    cf https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        The default values here are actually duplicates from the init function, but this was for readability (??)
        """
        parser = parent_parser.add_argument_group("keypointsDataset")
        parser.add_argument(
            "--image_dataset_path", required=False, type=str, help="path to the json file that defines the dataset"
        )
        parser.add_argument(
            "--json_dataset_path",
            required=False,
            type=str,
            help="path to the base dir from where the images are referenced in the json file of the dataset",
        )
        return parent_parser

    def __init__(
        self,
        json_dataset_path: str,
        image_dataset_path: str,
        keypoint_channels: Union[List[str], str],
        keypoint_channel_max_keypoints: Union[List[int], str],
        **kwargs,
    ):

        super(KeypointsDataset, self).__init__()
        self.json_file = json_dataset_path
        self.image_dir = image_dataset_path

        self.transform = ToTensor()  # convert images to Torch Tensors

        f = open(json_dataset_path, "r")

        # dataset format is defined in project README
        self.dataset = json.load(f)
        self.dataset = self.dataset["dataset"]

        if isinstance(keypoint_channels, list):
            self.keypoint_channels = keypoint_channels
        else:
            self.keypoint_channels = keypoint_channels.strip().split(" ")

        if isinstance(keypoint_channel_max_keypoints, list):
            self.keypoint_channel_max_keypoints = keypoint_channel_max_keypoints
        else:
            self.keypoint_channel_max_keypoints = [int(v) for v in keypoint_channel_max_keypoints.strip().split(" ")]

        # some checks on keypoint channels
        assert len(self.keypoint_channel_max_keypoints) == len(self.keypoint_channels)
        for channel in self.keypoint_channels:
            assert channel in self.dataset[0]

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def get_data_dir_path(cls) -> Path:
        return Path(__file__).resolve().parents[1] / "datasets"

    def convert_blender_coordinates_to_pixel_coordinates(
        self, keypoints: torch.Tensor, image_shape: int
    ) -> torch.Tensor:
        """
        Converts the keypoint coordinates as generated in Blender to discrete (u,v) pixel-coordinates
        with the origin in the top left corner and the u-axis going right.
        Note: only works for squared Images!

        Blender coords: [0,1] with v axis going up.
        """
        keypoints *= image_shape
        keypoints[:, 1] = image_shape - keypoints[:, 1]
        return keypoints

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image = self.get_image(index)
        image = self.transform(image)

        # read keypoints
        keypoints = []
        for i, keypoint_channel in enumerate(self.keypoint_channels):
            kp = torch.Tensor(self.dataset[index][keypoint_channel])
            kp = self.convert_blender_coordinates_to_pixel_coordinates(kp, image.shape[-1])

            # pad the tensor to make sure all items of a batch have same size and can hence be collated by the default
            # torch collate function.
            if self.keypoint_channel_max_keypoints[i] > 0:
                kp = pad_tensor_with_nans(kp, self.keypoint_channel_max_keypoints[i])

            keypoints.append(kp)

        return image, keypoints

    def get_image(self, index: int) -> np.ndarray:
        """
        read the image from disk and return as np array
        """
        # load images @runtime from disk
        image_path = os.path.join(os.getcwd(), self.image_dir, self.dataset[index]["image_path"])
        image = io.imread(image_path)
        return image


class KeypointsDatasetIOCatcher(KeypointsDataset):
    """
    This Dataset performs n attempts to load the dataset item, in an attempt
    to overcome IOErrors that can happen every now and then on a HPC.
     This does not require the entire dataset to be in memory.
    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return KeypointsDataset.add_argparse_args(parent_parser)

    def __init__(
        self,
        json_dataset_path: str,
        image_dataset_path: str,
        keypoint_channels: Union[List[str], str],
        keypoint_channel_max_keypoints: Union[List[int], str],
        n_io_attempts: int = 4,
        **kwargs,
    ):

        """
        n_io_attempts: number of trials to load image from IO
        """
        super().__init__(
            json_dataset_path, image_dataset_path, keypoint_channels, keypoint_channel_max_keypoints, **kwargs
        )
        self.n_io_attempts = n_io_attempts

    def get_image(self, index: int) -> np.ndarray:
        sleep_time_in_seconds = 1
        for j in range(self.n_io_attempts):
            try:
                image = super().get_image(index)  # IO read.
                return image
            except IOError:
                if j == self.n_io_attempts - 1:
                    raise IOError(f"Could not load image for dataset entry with index {index}")

                sleep_time = max(random.gauss(sleep_time_in_seconds, j), 0)
                print(
                    f"caught IOError in {j}th attempt to load image for item {index}, sleeping for {sleep_time} seconds"
                )
                time.sleep(sleep_time)
                sleep_time_in_seconds *= 2


class KeypointsDatasetPreloaded(KeypointsDatasetIOCatcher):
    """
    The images are preloaded in memory for faster access.
    This requires the whole dataset to fit into memory, so make sure to have enough memory available.

    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return KeypointsDatasetIOCatcher.add_argparse_args(parent_parser)

    def __init__(
        self,
        json_dataset_path: str,
        image_dataset_path: str,
        keypoint_channels: Union[List[str], str],
        keypoint_channel_max_keypoints: Union[List[int], str],
        **kwargs,
    ):

        """
        n_io_attempts: number of trials to load image from IO
        """
        super().__init__(
            json_dataset_path, image_dataset_path, keypoint_channels, keypoint_channel_max_keypoints, **kwargs
        )
        """
        n_io_attempts: number of trials to load image from IO
        """
        super().__init__(
            json_dataset_path, image_dataset_path, keypoint_channels, keypoint_channel_max_keypoints, **kwargs
        )
        self.preloaded_images = [None] * len(self.dataset)
        self._preload()

    def _preload(self):
        """
        load images into memory as np.ndarrays.
        Choice to load them as np.ndarrays is because pytorch uses float32 for each value whereas
        the original values are only 8 bit ints, so this is a 4 times increase in size..
        """

        print("preloading dataset images")
        for i in tqdm.trange(len(self)):
            self.preloaded_images[i] = super().get_image(i)
        print("dataset images preloaded")

    def get_image(self, index: int) -> np.ndarray:
        return self.preloaded_images[index]
