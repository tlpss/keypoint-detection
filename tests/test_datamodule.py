import os
import unittest

import torch

from keypoint_detection.data.datamodule import RandomSplitDataModule
from keypoint_detection.data.dataset import KeypointsDataset


class TestDataModule(unittest.TestCase):
    def setUp(self):
        TEST_DIR = os.path.dirname(os.path.abspath(__file__))
        self.json_path = os.path.join(TEST_DIR, "test_dataset/dataset.json")
        self.image_path = os.path.join(TEST_DIR, "test_dataset")
        self.dataset = KeypointsDataset(
            self.json_path, self.image_path, "corner_keypoints flap_corner_keypoints", "-1 8 "
        )

    def test_split(self):
        module = RandomSplitDataModule(self.dataset, 1, 0.0, 2)
        train_dataloader = module.train_dataloader()
        self.assertEqual(len(train_dataloader), 4)

        module = RandomSplitDataModule(self.dataset, 1, 0.5, 2)
        train_dataloader = module.train_dataloader()
        validation_dataloader = module.train_dataloader()
        self.assertEqual(len(train_dataloader), 2)
        self.assertEqual(len(validation_dataloader), 2)

    def test_batch_format(self):
        module = RandomSplitDataModule(self.dataset, 2, 0.0, 2)
        train_dataloader = module.train_dataloader()

        batch = next(iter(train_dataloader))
        self.assertEqual(len(batch), 2)

        img, keypoints = batch

        corner_kp, flap_kp = keypoints

        self.assertIsInstance(img, torch.Tensor)
        self.assertEquals(img.shape, (2, 3, 64, 64))
        self.assertIsInstance(corner_kp, torch.Tensor)
        self.assertEquals(corner_kp.shape, (2, 4, 3))
        self.assertIsInstance(flap_kp, torch.Tensor)
        self.assertEquals(flap_kp.shape, (2, 8, 3))
