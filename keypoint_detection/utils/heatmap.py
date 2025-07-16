import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from skimage.feature import peak_local_max



import torch
import torch.nn.functional as F

def focal_loss_with_logits(
    input: torch.Tensor, 
    target: torch.Tensor, 
    alpha: float = 0.25, 
    gamma: float = 2.0, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Focal Loss implementation for binary classification with logits.
    
    This addresses class imbalance by down-weighting easy examples and focusing
    on hard examples. Particularly useful for keypoint detection where most
    pixels are background.
    
    Args:
        input: Predicted logits (before sigmoid), shape [N, H, W] or [N, C, H, W]
        target: Ground truth binary targets, shape [N, H, W] or [N, C, H, W]
        alpha: Weighting factor for rare class (keypoints). Range [0, 1]
               alpha=0.25 means 25% weight for positive class, 75% for negative
        gamma: Focusing parameter. Higher gamma = more focus on hard examples
               gamma=0 reduces to standard cross-entropy
               gamma=2 is typical default
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Focal loss tensor
    """
    # Convert logits to probabilities
    p = torch.sigmoid(input)
    
    # Calculate cross entropy
    ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    
    # Calculate p_t for focal weight
    p_t = p * target + (1 - p) * (1 - target)
    
    # Calculate alpha weight
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    # Calculate focal weight
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    # Apply focal weight
    focal_loss = focal_weight * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def adaptive_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Adaptive focal loss that automatically adjusts alpha based on the 
    positive/negative ratio in the current batch.
    
    Useful when the class imbalance varies between batches.
    """
    # Calculate positive ratio in current batch
    positive_ratio = target.mean().item()
    
    # Adjust alpha based on positive ratio
    # If few positives, increase alpha (give more weight to positives)
    if positive_ratio < 0.1:
        alpha = 0.75
    elif positive_ratio < 0.3:
        alpha = 0.5
    else:
        alpha = 0.25
    
    return focal_loss_with_logits(input, target, alpha, gamma, reduction)



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


def get_keypoints_from_heatmap_scipy(
    heatmap: torch.Tensor, min_keypoint_pixel_distance: int, max_keypoints: int = 20
) -> List[Tuple[int, int]]:
    """
    Extracts at most 20 keypoints from a heatmap, where each keypoint is defined as being a local maximum within a 2D mask [ -min_pixel_distance, + pixel_distance]^2
    cf https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max

    THIS IS SLOW! use get_keypoints_from_heatmap_batch_maxpool instead.


    Args:
        heatmap : heatmap image
        min_keypoint_pixel_distance : The size of the local mask, serves as NMS
        max_keypoints: the amount of keypoints to determine from the heatmap, -1 to return all points. Defaults to 20 to limit computational burder
        for models that predict random keypoints in early stage of training.

    Returns:
        A list of 2D keypoints
    """
    warnings.warn("get_keypoints_from_heatmap_scipy is slow! Use get_keypoints_from_heatmap_batch_maxpool instead.")
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


def get_keypoints_from_heatmap_batch_maxpool(
    heatmap: torch.Tensor,
    max_keypoints: int = 20,
    min_keypoint_pixel_distance: int = 1,
    abs_max_threshold: Optional[float] = None,
    rel_max_threshold: Optional[float] = None,
    return_scores: bool = False,
) -> List[List[List[Tuple[int, int]]]]:
    """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

    Inspired by mmdetection and CenterNet:
      https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html

    Args:
        heatmap (torch.Tensor): NxCxHxW heatmap batch
        max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
        min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

        Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
        abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
        rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

    Returns:
        The extracted keypoints for each batch, channel and heatmap; and their scores
    """

    # TODO: maybe separate the thresholding into another function to make sure it is not used during training, where it should not be used?

    # TODO: ugly that the output can change based on a flag.. should always return scores and discard them when I don't need them...

    batch_size, n_channels, _, width = heatmap.shape

    # obtain max_keypoints local maxima for each channel (w/ maxpool)

    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = min_keypoint_pixel_distance
    # exclude border keypoints by padding with highest possible value
    # bc the borders are more susceptible to noise and could result in false positives
    padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
    max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)
    # if the value equals the original value, it is the local maximum
    local_maxima = max_pooled_heatmap == heatmap
    # all values to zero that are not local maxima
    heatmap = heatmap * local_maxima

    # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
    scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
    indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
    # at this point either score > 0.0, in which case the index is a local maximum
    # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

    #  remove top-k that are not local maxima and threshold (if required)
    # thresholding shouldn't be done during training

    #  moving them to CPU now to avoid multiple GPU-mem accesses!
    indices = indices.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    # determine NMS threshold
    threshold = 0.01  # make sure it is > 0 to filter out top-k that are not local maxima
    if abs_max_threshold is not None:
        threshold = max(threshold, abs_max_threshold)
    if rel_max_threshold is not None:
        threshold = max(threshold, rel_max_threshold * heatmap.max())

    # have to do this manually as the number of maxima for each channel can be different
    for batch_idx in range(batch_size):
        for channel_idx in range(n_channels):
            candidates = indices[batch_idx, channel_idx]
            for candidate_idx in range(candidates.shape[0]):

                # these are filtered out directly.
                if scores[batch_idx, channel_idx, candidate_idx] > threshold:
                    # convert to (u,v)
                    filtered_indices[batch_idx][channel_idx].append(candidates[candidate_idx][::-1].tolist())
                    filtered_scores[batch_idx][channel_idx].append(scores[batch_idx, channel_idx, candidate_idx])
    if return_scores:
        return filtered_indices, filtered_scores
    else:
        return filtered_indices


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


if __name__ == "__main__":
    import torch.profiler as profiler

    keypoints = torch.tensor([[150, 134], [64, 153]]).cuda()
    heatmap = generate_channel_heatmap((1080, 1920), keypoints, 6, "cuda")
    heatmap = heatmap.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
    # heatmap = torch.stack([heatmap, heatmap], dim=0)
    print(heatmap.shape)
    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("get_keypoints_from_heatmap_batch_maxpool"):
            print(get_keypoints_from_heatmap_batch_maxpool(heatmap, 50, min_keypoint_pixel_distance=5))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
