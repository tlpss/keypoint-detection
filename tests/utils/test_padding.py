import unittest

import torch

from keypoint_detection.utils.tensor_padding import pad_tensor_with_nans, unpad_nans_from_tensor


class TestPadding(unittest.TestCase):
    def test_padd(self):
        tensor = torch.tensor([[1.0, 2, 3], [3.2, 4, 5]])

        padded_tensor = pad_tensor_with_nans(tensor, 4)
        self.assertEqual(padded_tensor.shape, (4, 3))
        unpadded_tensor = unpad_nans_from_tensor(padded_tensor)
        self.assertAlmostEqual(torch.sum(tensor - unpadded_tensor), 0)

    def test_pad_full_size_tensor(self):
        tensor = torch.tensor([[1.0, 2, 3], [3.2, 4, 5]])

        padded_tensor = pad_tensor_with_nans(tensor, 2)
        self.assertEqual(padded_tensor.shape, (2, 3))
        unpadded_tensor = unpad_nans_from_tensor(padded_tensor)
        self.assertAlmostEqual(torch.sum(tensor - unpadded_tensor), 0)

    def test_batch_unpadding(self):
        tensor1 = torch.tensor([[1.0, 2, 3], [3.2, 4, 5]])
        tensor2 = torch.tensor([[1.0, 2, 3]])

        tensor = torch.stack([pad_tensor_with_nans(tensor1, 2), pad_tensor_with_nans(tensor2, 2)])
        unpadded = [unpad_nans_from_tensor(tensor[i, :]) for i in range(tensor.shape[0])]

        self.assertTrue(torch.equal(tensor1, unpadded[0]))
        self.assertTrue(torch.equal(tensor2, unpadded[1]))
