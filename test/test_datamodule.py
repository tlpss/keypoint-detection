import unittest

import torch

from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.data.datamodule import KeypointsDataModule

from .configuration import DEFAULT_HPARAMS, TEST_PARAMS


class TestDataModule(unittest.TestCase):
    def setUp(self):
        self.dataset = COCOKeypointsDataset(**DEFAULT_HPARAMS)

    def test_split(self):
        size = TEST_PARAMS["dataset_size"]
        for train_ratio in [1.0, 0.5]:
            module = KeypointsDataModule(self.dataset, 1, 1 - train_ratio, 2)
            train_dataloader = module.train_dataloader()
            self.assertEqual(len(train_dataloader), train_ratio * size)
            self.assertEqual(len(module.val_dataloader()), (1 - train_ratio) * size)

    def test_batch_format(self):
        batch_size = 3
        module = KeypointsDataModule(self.dataset, batch_size, 0.0, 2)
        train_dataloader = module.train_dataloader()

        batch = next(iter(train_dataloader))
        self.assertEqual(len(batch), 2)

        img, keypoints = batch

        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (batch_size, 3, 64, 64))

        # check order of lists: channels, batch
        self.assertTrue(isinstance(keypoints, list))
        n_channels = len(DEFAULT_HPARAMS["keypoint_channel_configuration"])
        self.assertEqual(len(keypoints), n_channels)
        for i in range(n_channels):
            self.assertTrue(isinstance(keypoints[i], list))
            self.assertEqual(len(keypoints[i]), batch_size)

        ch1, ch2 = keypoints

        self.assertIsInstance(ch1[0], torch.Tensor)
        self.assertIsInstance(ch2[0], torch.Tensor)
