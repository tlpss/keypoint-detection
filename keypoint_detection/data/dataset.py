import argparse
import json
from pathlib import Path
from typing import List, Union

import torch
from torchvision.transforms import ToTensor

from keypoint_detection.data.utils import ImageDataset, ImageLoader, IOSafeImageLoaderDecorator
from keypoint_detection.utils.tensor_padding import pad_tensor_with_nans


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
            "--image_dataset_path",
            required=False,
            type=str,
            help="Absolute path to the json file that defines the dataset according to the defined format.",
        )
        parser.add_argument(
            "--json_dataset_path",
            required=False,
            type=str,
            help="Absolute path to the base dir from where the images are referenced in the json file of the dataset.",
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

        image_loader = IOSafeImageLoaderDecorator(ImageLoader())
        super(KeypointsDataset, self).__init__(image_loader)
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

        image_path = Path(".") / self.image_dir / self.dataset[index]["image_path"]
        image = self.image_loader.get_image(image_path, index)
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
