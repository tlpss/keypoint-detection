from pathlib import Path
from typing import List, Union

import torch

from keypoint_detection.data.dataset import KeypointsDataset


class UnlabeledKeypointsDataset(KeypointsDataset):
    """
    Simple dataset to run inference on unlabeled data
    """

    def __init__(
        self,
        json_dataset_path: str,
        image_dataset_path: str,
        keypoint_channels: Union[List[str], str],
        keypoint_channel_max_keypoints: Union[List[int], str],
        **kwargs,
    ):

        super().__init__(
            json_dataset_path, image_dataset_path, keypoint_channels, keypoint_channel_max_keypoints, **kwargs
        )

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)
        image_path = Path(".") / self.image_dir / self.dataset[index]["image_path"]

        image = self.image_loader.get_image(image_path, index)
        image = self.transform(image)

        return image
