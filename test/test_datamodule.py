import copy
import unittest

import torch

from keypoint_detection.data.datamodule import KeypointsDataModule

from .configuration import DEFAULT_HPARAMS, TEST_PARAMS

import random 

class TestDataModule(unittest.TestCase):
    def test_split(self):
        size = TEST_PARAMS["dataset_size"]
        for train_ratio in [1.0, 0.5]:
            HPARAMS = copy.deepcopy(DEFAULT_HPARAMS)
            HPARAMS["batch_size"] = 1
            HPARAMS["validation_split_ratio"] = 1 - train_ratio
            module = KeypointsDataModule(**HPARAMS)
            train_dataloader = module.train_dataloader()
            self.assertEqual(len(train_dataloader), train_ratio * size)
            self.assertEqual(len(module.val_dataloader()), (1 - train_ratio) * size)

    def test_batch_format(self):
        batch_size = 2

        hparams = copy.deepcopy(DEFAULT_HPARAMS)
        hparams["batch_size"] = batch_size
        hparams["validation_split_ratio"] = 0.0
        module = KeypointsDataModule(**hparams)
        train_dataloader = module.train_dataloader()

        batch = next(iter(train_dataloader))
        self.assertEqual(len(batch), batch_size)

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

    def test_augmentations_result_in_different_image(self):
        random.seed(2022)
        hparams = copy.deepcopy(DEFAULT_HPARAMS)
        module = KeypointsDataModule(**hparams)
        train_dataloader = module.train_dataloader()

        batch = next(iter(train_dataloader))
        img, _ = batch

        hparams = copy.deepcopy(DEFAULT_HPARAMS)
        hparams["augment_train"] = True
        module = KeypointsDataModule(**hparams)
        train_dataloader = module.train_dataloader()

        batch = next(iter(train_dataloader))
        transformed_img, _ = batch
        # check both images are not equal.
        self.assertTrue(torch.linalg.norm(img-transformed_img))
