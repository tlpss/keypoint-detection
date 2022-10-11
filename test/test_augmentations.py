import unittest

import albumentations as A
import numpy as np

from keypoint_detection.data.augmentations import MultiChannelKeypointsCompose


class TestRandomCrop(unittest.TestCase):
    def setUp(self) -> None:
        self.keypoints = [[[20, 30]], [[100, 100], [255, 255]]]
        self.img = np.zeros((256, 256, 3))

    def test_random_crop_keypoint_ordering(self):
        transform = MultiChannelKeypointsCompose([A.Crop(x_max=100, y_max=100)])
        transformed = transform(image=self.img, keypoints=self.keypoints)
        image, ordered_keypoints = transformed["image"], transformed["keypoints"]
        self.assertEqual(len(ordered_keypoints), 2)
        self.assertEqual(len(ordered_keypoints[0]), 1)
        self.assertEqual(len(ordered_keypoints[1]), 0)
        self.assertEqual(image.shape, (100, 100, 3))
