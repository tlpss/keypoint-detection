from typing import List, Tuple

import numpy as np
import torch


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
    heatmap: torch.Tensor, min_keypoint_pixel_distance: int, max_keypoints: int = 20, threshold_abs=None, threshold_rel=None
) -> List[Tuple[int, int]]:
    """
    Extracts at most 20 keypoints from a heatmap, where each keypoint is defined as being a local maximum within a 2D mask [ -min_pixel_distance, + pixel_distance]^2
    Originally https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max was used,
    but a simpler and more efficient algorithm was designed which could keep all data on the GPU. However, the functionality is not 100% the same.
    This implementation is a simple filter and non-max supression step based on the original Skimage implementation.
    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.
    Args:
        heatmap : heatmap image
        min_keypoint_pixel_distance : The size of the local mask
        max_keypoints: the amount of keypoints to determine from the heatmap, -1 to return all points. Defaults to 20 to limit computational burder
        for models that predict random keypoints in early stage of training.
        threshold_abs : Minimum intensity of peaks. By default (None), the absolute threshold is the minimum intensity of the image.
        threshold_rel : Minimum intensity of peaks, calculated as ``max(image) * threshold_rel``.
    Returns:
        A tensor of 2D keypoints
    """
    # Set the correct threshold
    threshold = threshold_abs if threshold_abs is not None else heatmap.min()
    if threshold_rel is not None:
        threshold = max(threshold, threshold_rel * heatmap.max())

    output_points = torch.empty((0, 2), dtype=float, device=heatmap.device)
    nms_range_squared = min_keypoint_pixel_distance ** 2

    kpt_candidates_x, kpt_candidates_y = torch.where(heatmap > threshold)
    kpt_candidates = torch.vstack((kpt_candidates_x, kpt_candidates_y)).T
    kpt_candidates_scores = heatmap[heatmap > threshold]
    order = torch.argsort(kpt_candidates_scores, descending=True)
    kpt_candidates = kpt_candidates[order]

    # From highest to lowest score
    while kpt_candidates.shape[0] > 0:
        # Add the current keypoint and remove those within the NMS range until none are left
        kpt = kpt_candidates[0]
        output_points = torch.vstack((output_points, kpt))

        distances = torch.pow(kpt_candidates - kpt, 2).sum(dim=1)

        # Identify points outside of NMS range and keep them
        keypoints_outside_nms_mask = distances > nms_range_squared
        kpt_candidates = kpt_candidates[keypoints_outside_nms_mask]
    
    # Make sure the points are ordered X, Y
    output_points = output_points.permute(1, 0)

    if max_keypoints > 0:
        # The keypoints are added in decreasing score so we can just take the top k
        output_points = output_points[:max_keypoints]

    return output_points


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
