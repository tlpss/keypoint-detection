import unittest

import torch

from keypoint_detection.utils.heatmap import gaussian_heatmap, generate_keypoints_heatmap, get_keypoints_from_heatmap


class TestHeatmapUtils(unittest.TestCase):
    def setUp(self):
        self.image_width = 32
        self.image_height = 16
        self.keypoints = [[10, 4], [10, 8], [30, 7]]
        self.sigma = 3

    def test_gaussian(self):
        img = gaussian_heatmap(
            (self.image_height, self.image_width), self.keypoints[0], torch.Tensor([self.sigma]), device="cpu"
        )
        self.assertEqual(img.shape, (self.image_height, self.image_width))

    def test_keypoint_generation_and_extraction(self):
        heatmap = generate_keypoints_heatmap((self.image_height, self.image_width), self.keypoints, self.sigma, "cpu")
        extracted_keypoints = get_keypoints_from_heatmap(heatmap, 1)
        for keypoint in extracted_keypoints:
            self.assertTrue(keypoint in self.keypoints)
        self.assertEqual(self.image_height, heatmap.shape[0])
        self.assertEqual(self.image_width, heatmap.shape[1])
