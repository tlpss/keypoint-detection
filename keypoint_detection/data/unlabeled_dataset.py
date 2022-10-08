import os

import torch
from torchvision.transforms import ToTensor

from keypoint_detection.data.imageloader import ImageDataset


class UnlabeledKeypointsDataset(ImageDataset):
    """
    Simple dataset to run inference on unlabeled data
    """

    def __init__(
        self,
        image_dataset_path: str,
        **kwargs,
    ):
        super().__init__()
        self.image_paths = os.listdir(image_dataset_path)
        self.image_paths = [image_dataset_path + f"/{path}" for path in self.image_paths]

        self.transform = ToTensor()  # convert images to Torch Tensors

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image_path = self.image_paths[index]
        image = self.image_loader.get_image(image_path, index)
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)
