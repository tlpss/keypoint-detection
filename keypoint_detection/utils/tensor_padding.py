import numpy as np
import torch


def pad_tensor_with_nans(tensor: torch.Tensor, desired_length) -> torch.Tensor:
    assert len(tensor.shape) == 2
    l, w = tensor.shape
    nan_tensor = torch.tensor([[np.NAN] * w] * (desired_length - l))
    padded_tensor = torch.cat([tensor, nan_tensor])
    return padded_tensor


def unpad_nans_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    assert len(tensor.shape) == 2
    filtered_tensor = tensor[~torch.all(tensor.isnan(), dim=1)]
    return filtered_tensor
