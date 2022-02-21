import unittest

import torch

from keypoint_detection.data.dataset import KeypointsDataset

from .configuration import DEFAULT_HPARAMS


class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.hparams = DEFAULT_HPARAMS

    def test_dataset(self):
        dataset = KeypointsDataset(**self.hparams)
        item = dataset.__getitem__(0)
        img, keypoints = item

        self.assertEqual(len(keypoints), 2)
        corner, flap = keypoints

        self.assertEqual(img.shape, (3, 64, 64))
        self.assertEqual(corner.shape, (4, 3))
        self.assertEqual(flap.shape, (8, 3))

        self.assertTrue(isinstance(img, torch.Tensor))
        self.assertTrue(isinstance(corner, torch.Tensor))
