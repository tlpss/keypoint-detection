import unittest

import torch

from keypoint_detection.data.blender_dataset import BlenderKeypointsDataset
from keypoint_detection.data.unlabeled_dataset import UnlabeledKeypointsDataset

from .configuration import DEFAULT_HPARAMS


class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.hparams = DEFAULT_HPARAMS

    def test_dataset(self):
        dataset = BlenderKeypointsDataset(**self.hparams)
        item = dataset.__getitem__(0)
        img, keypoints = item

        self.assertEqual(len(keypoints), 2)
        corner, flap = keypoints

        self.assertEqual(img.shape, (3, 64, 64))
        self.assertEqual(corner.shape, (4, 3))
        self.assertEqual(flap.shape, (8, 3))

        self.assertTrue(isinstance(img, torch.Tensor))
        self.assertTrue(isinstance(corner, torch.Tensor))


class TestUnlabeledDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = UnlabeledKeypointsDataset(DEFAULT_HPARAMS["image_dataset_path"] + "/images")

        self.assertEqual(len(dataset), 4)
        img = dataset[0]
        self.assertEqual(img.shape, (3, 64, 64))
