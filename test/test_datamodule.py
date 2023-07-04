import copy
import random
import unittest
from test.configuration import DEFAULT_HPARAMS, TEST_PARAMS

import torch
import torch.utils.data

from keypoint_detection.data.datamodule import KeypointsDataModule


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
        # get the dataset through the datamodule
        # cannot use dataloader directly bc it shuffles the dataset.
        random.seed(2022)
        torch.manual_seed(2022)

        hparams = copy.deepcopy(DEFAULT_HPARAMS)
        hparams["augment_train"] = False

        module = KeypointsDataModule(**hparams)
        no_aug_train_dataloader = module.train_dataloader()
        no_aug_dataset = no_aug_train_dataloader.dataset

        img, _ = no_aug_dataset[0]

        # reset seeds to obtain the same dataset order
        # and get the dataset again but now with augmentations
        random.seed(2022)
        torch.manual_seed(2022)
        hparams = copy.deepcopy(hparams)
        hparams["augment_train"] = True
        aug_module = KeypointsDataModule(**hparams)
        aug_train_dataloader = aug_module.train_dataloader()
        aug_dataset = aug_train_dataloader.dataset

        dissimilar_images = 0
        # iterate a few times over the dataset to check that the augmentations are applied
        # bc none of the augmentations is applied with 100% probability so some batches could be equal
        # and finding a seed that triggers them could change if you change the augmentations
        for _ in range(5):
            transformed_img, _ = aug_dataset[0]
            # check both images are not equal.
            dissimilar_images += 1 * (torch.linalg.norm(img - transformed_img) != 0.0)

        self.assertTrue(dissimilar_images > 0)
