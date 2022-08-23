import unittest

import torch

from keypoint_detection.utils.heatmap import generate_channel_heatmap, get_keypoints_from_heatmap, create_heatmap_batch


class TestHeatmapUtils(unittest.TestCase):
    def setUp(self):
        self.image_width = 32
        self.image_height = 16
        self.keypoints = torch.Tensor([[10, 4], [10, 8], [30, 7]])
        self.sigma = 3.1

    def test_keypoint_generation_and_extraction(self):
        # test if extract(generate(keypoints)) == keypoints 
        heatmap = generate_channel_heatmap((self.image_height, self.image_width), self.keypoints, self.sigma, "cpu")
        extracted_keypoints = get_keypoints_from_heatmap(heatmap, 1)
        for keypoint in extracted_keypoints:
            self.assertTrue(keypoint in self.keypoints.tolist())
        self.assertEqual((self.image_height, self.image_width), heatmap.shape)
        self.assertGreater(heatmap[4,10],0.5)

    
    def test_empty_heatmap(self):
        # test if heatmap for channel w/o keypoints is created correctly
        heatmap = generate_channel_heatmap((self.image_height, self.image_width),torch.tensor([[]]),self.sigma, "cpu")
        self.assertEqual(self.image_height, heatmap.shape[0])
        self.assertEqual(self.image_width, heatmap.shape[1])
        self.assertAlmostEqual(torch.max(heatmap),0.0)


    def test_heatmap_batch(self):
        batch_channel_list_of_keypoint_tensors =[self.keypoints,self.keypoints, torch.Tensor([[]])]
        batch_heatmap = create_heatmap_batch((self.image_height, self.image_width), batch_channel_list_of_keypoint_tensors,self.sigma, "cpu")
        self.assertEqual(batch_heatmap.shape, (len(batch_channel_list_of_keypoint_tensors), self.image_height, self.image_width))
