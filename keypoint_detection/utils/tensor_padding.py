import numpy as np
import torch


def pad_tensor_with_nans(tensor: torch.Tensor, desired_length: int) -> torch.Tensor:
    """ pads (possibly zero-sized) tensor to a desired_length x 2 tensor.

    Args:
        tensor (torch.Tensor): _description_
        desired_length (int): _description_

    Returns:
        torch.Tensor: desired_length x Tensor
    """

    if tensor.numel() == 0: #empty Tensor
        return torch.tensor([[np.NaN]* 2] * desired_length)
    
    assert len(tensor.shape) == 2
    l, w = tensor.shape
    assert w == 2
    nan_tensor = torch.tensor([[np.NAN] * w] * (desired_length - l))
    padded_tensor = torch.cat([tensor, nan_tensor])
    return padded_tensor


def unpad_nans_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    assert len(tensor.shape) == 2
    filtered_tensor = tensor[~torch.all(tensor.isnan(), dim=1)]
    return filtered_tensor

if __name__ == "__main__":
    t = pad_tensor_with_nans(torch.Tensor([]),4)
    l = unpad_nans_from_tensor(t)
    assert (l.numel() == 0)

    x = torch.randn(4,2)
    t = pad_tensor_with_nans(x,8)
    l = unpad_nans_from_tensor(t)
    assert(l.shape == (4,2))
    assert torch.equal(x,l)