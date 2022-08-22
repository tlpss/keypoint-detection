import unittest

import torch

from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.data.unlabeled_dataset import UnlabeledKeypointsDataset

from .configuration import DEFAULT_HPARAMS, TEST_PARAMS


class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.hparams = DEFAULT_HPARAMS

    def test_dataset(self):
        dataset = COCOKeypointsDataset(**self.hparams)

        self.assertEqual(len(dataset),TEST_PARAMS["dataset_size"])
        item = dataset[0]
        img, keypoints = item

        self.assertEqual(len(keypoints), len(DEFAULT_HPARAMS["keypoint_channels"]))
        ch1,ch2 = keypoints

        # check img 1 
        self.assertEqual(img.shape, (3, TEST_PARAMS["image_size"], TEST_PARAMS["image_size"]))
        self.assertEqual(len(ch1),len(DEFAULT_HPARAMS["keypoint_channels"][0]))
        self.assertEqual(len(ch2), len(DEFAULT_HPARAMS["keypoint_channels"][1]))

        # check img 2,which has duplicates in channel 1 
        item = dataset[1]
        img, keypoints = item
        self.assertEqual(len(keypoints), len(DEFAULT_HPARAMS["keypoint_channels"]))
        ch1,ch2 = keypoints
        print(ch1)
        self.assertEqual(img.shape, (3, 64, 64))
        self.assertEqual(len(ch1),2*len(DEFAULT_HPARAMS["keypoint_channels"][0]))
        self.assertEqual(len(ch2), len(DEFAULT_HPARAMS["keypoint_channels"][1]))

    def test_non_visible_dataset(self):
        self.hparams.update({"detect_non_visible_keypoints": False})
        dataset = COCOKeypointsDataset(**self.hparams)

        # has duplicates but they are not visible (flag=1)
        item = dataset[1]
        img, keypoints = item
        ch1,ch2 = keypoints
        self.assertEqual(len(ch1),len(DEFAULT_HPARAMS["keypoint_channels"][0]))
        self.assertEqual(len(ch2), len(DEFAULT_HPARAMS["keypoint_channels"][1]))


class TestUnlabeledDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = UnlabeledKeypointsDataset(DEFAULT_HPARAMS["image_dataset_path"])

        self.assertEqual(len(dataset), 4)
        img = dataset[0]
        self.assertEqual(img.shape, (3, 64, 64))
