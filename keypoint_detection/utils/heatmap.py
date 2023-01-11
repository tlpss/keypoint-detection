from typing import List, Tuple

import numpy as np
import torch
from skimage.feature import peak_local_max


def BCE_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple BCE loss. Used to compute the BCE of the ground truth heatmaps as the BCELoss in Pytorch complains
    about instability in FP16 (even with no_grad).
    """
    return -(
        target * torch.clip(torch.log(input + 1e-10), -100, 100)
        + (1 - target) * torch.clip(torch.log(1 - input + 1e-10), -100, 100)
    ).mean()


def create_heatmap_batch(
    shape: Tuple[int, int], keypoints: List[torch.Tensor], sigma: float, device: torch.device
) -> torch.Tensor:
    """[summary]

    Args:
        shape (Tuple): H,W
        keypoints (List[torch.Tensor]): N Tensors of size K_i x 2  with batch of keypoints.

    Returns:
        (torch.Tensor): N x H x W Tensor with N heatmaps
    """

    batch_heatmaps = [generate_channel_heatmap(shape, keypoints[i], sigma, device) for i in range(len(keypoints))]
    batch_heatmaps = torch.stack(batch_heatmaps, dim=0)
    return batch_heatmaps


def generate_channel_heatmap(
    image_size: Tuple[int, int], keypoints: torch.Tensor, sigma: float, device: torch.device
) -> torch.Tensor:
    """
    Generates heatmap with gaussian blobs for each keypoint, using the given sigma.
    Max operation is used to combine the heatpoints to avoid local optimum surpression.
    Origin is topleft corner and u goes right, v down.

    Args:
        image_size: Tuple(int,int) that specify (H,W) of the heatmap image
        keypoints: a 2D Tensor K x 2,  with K keypoints  (u,v).
        sigma: (float) std deviation of the blobs
        device: the device on which to allocate new tensors

    Returns:
         Torch.tensor:  A Tensor with the combined heatmaps of all keypoints.
    """

    # cast keypoints (center) to ints to make grid align with pixel raster.
    #  Otherwise, the AP metric for  d = 1 will not result in 1
    #  if the gt_heatmaps are used as input.

    assert isinstance(keypoints, torch.Tensor)

    if keypoints.numel() == 0:
        # special case for which there are no keypoints in this channel.
        return torch.zeros(image_size, device=device)

    u_axis = torch.linspace(0, image_size[1] - 1, image_size[1], device=device)
    v_axis = torch.linspace(0, image_size[0] - 1, image_size[0], device=device)
    # create grid values in 2D with x and y coordinate centered aroud the keypoint
    v_grid, u_grid = torch.meshgrid(v_axis, u_axis, indexing="ij")  # v-axis -> dim 0, u-axis -> dim 1

    u_grid = u_grid.unsqueeze(0) - keypoints[..., 0].unsqueeze(-1).unsqueeze(-1)
    v_grid = v_grid.unsqueeze(0) - keypoints[..., 1].unsqueeze(-1).unsqueeze(-1)

    ## create gaussian around the centered 2D grids $ exp ( -0.5 (x**2 + y**2) / sigma**2)$
    heatmap = torch.exp(
        -0.5 * (torch.square(u_grid) + torch.square(v_grid)) / torch.square(torch.tensor([sigma], device=device))
    )
    heatmap = torch.max(heatmap, dim=0)[0]
    return heatmap


def get_keypoints_from_heatmap(
    heatmap: torch.Tensor, min_keypoint_pixel_distance: int, max_keypoints: int = 20
) -> List[Tuple[int, int]]:
    """
    Extracts at most 20 keypoints from a heatmap, where each keypoint is defined as being a local maximum within a 2D mask [ -min_pixel_distance, + pixel_distance]^2
    cf https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max

    Args:
        heatmap : heatmap image
        min_keypoint_pixel_distance : The size of the local mask
        max_keypoints: the amount of keypoints to determine from the heatmap, -1 to return all points. Defaults to 20 to limit computational burder
        for models that predict random keypoints in early stage of training.

    Returns:
        A list of 2D keypoints
    """

    np_heatmap = heatmap.cpu().numpy().astype(np.float32)

    # num_peaks and rel_threshold are set to limit computational burden when models do random predictions.
    max_keypoints = max_keypoints if max_keypoints > 0 else np.inf
    keypoints = peak_local_max(
        np_heatmap,
        min_distance=min_keypoint_pixel_distance,
        threshold_rel=0.1,
        threshold_abs=0.1,
        num_peaks=max_keypoints,
    )

    return keypoints[::, ::-1].tolist()  # convert to (u,v) aka (col,row) coord frame from (row,col)


def compute_keypoint_probability(heatmap: torch.Tensor, detected_keypoints: List[Tuple[int, int]]) -> List[float]:
    """Compute probability measure for each detected keypoint on the heatmap

    Args:
        heatmap: Heatmap
        detected_keypoints: List of extreacted keypoints

    Returns:
        : [description]
    """
    # note the order! (u,v) is how we write , but the heatmap has to be indexed (v,u) as it is H x W
    return [heatmap[k[1]][k[0]].item() for k in detected_keypoints]
