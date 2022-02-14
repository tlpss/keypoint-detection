from typing import List, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from matplotlib import cm
from PIL import Image
from skimage.feature import peak_local_max


def gaussian_heatmap(
    image_size: Tuple[int, int],
    center: Union[Tuple[float, float], List[float]],
    sigma: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Creates a Gaussian blob heatmap for a single keypoint.
    The coordinate system is a left-top corner origin with u going left and v going down.

    Args:
        image_size (Tuple(int,int)): image_size (height, width) ! note convention to match store-order and tensor dimensions
        center (Tuple(int,int)): center coordinate (cX,cY) (U,V) ! note: U,V order
        sigma (torch.Tensor): standard deviation of the gaussian blob

    Returns:
        Torch.Tensor: A tensor with zero background, specified size and a Gaussian heatmap around the center.
    """

    # cast keypoints (center) to ints to make grid align with pixel raster.
    #  Otherwise, the AP metric for  d = 1 will not result in 1
    #  if the gt_heatmaps are used as input.
    u_axis = torch.linspace(0, image_size[1] - 1, image_size[1], device=device) - int(center[0])
    v_axis = torch.linspace(0, image_size[0] - 1, image_size[0], device=device) - int(center[1])
    # create grid values in 2D with x and y coordinate centered aroud the keypoint
    xx, yy = torch.meshgrid(v_axis, u_axis)
    ## create gaussian around the centered 2D grids $ exp ( -0.5 (x**2 + y**2) / sigma**2)$
    heatmap = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sigma))
    return heatmap


def generate_keypoints_heatmap(
    image_size: Tuple[int, int], keypoints: List[Tuple[float, float]], sigma: float, device: torch.device
) -> torch.Tensor:
    """
    Generates heatmap with gaussian blobs for each keypoint, using the given sigma.
    Max operation is used to combine the heatpoints to avoid local optimum surpression.
    Origin is topleft corner and u goes right, v down.

    Args:
        image_size: Tuple(int,int) that specify (H,W) of the heatmap image
        keypoints: List(Tuple(int,int), ...) with keypoints (u,v).
        sigma: (float) std deviation of the blobs
        device: the device on which to allocate new tensors

    Returns:
         Torch.tensor:  A Tensor with the combined heatmaps of all keypoints.
    """

    img = torch.zeros(image_size, device=device)  # (h,w) dimensions
    sigma = torch.Tensor([sigma]).to(device)
    for keypoint in keypoints:
        new_img = gaussian_heatmap(image_size, keypoint, sigma, device)
        img = torch.maximum(img, new_img)  # piecewise max of 2 Tensors
    return img


def get_keypoints_from_heatmap(
    heatmap: torch.Tensor, min_keypoint_pixel_distance: int, max_keypoints=None
) -> List[Tuple[int, int]]:
    """
    Extracts at most 20 keypoints from a heatmap, where each keypoint is defined as being a local maximum within a 2D mask [ -min_pixel_distance, + pixel_distance]^2
    cf https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max

    Args:
        heatmap : heatmap image
        min_keypoint_pixel_distance : The size of the local mask

    Returns:
        A list of 2D keypoints
    """

    np_heatmap = heatmap.cpu().numpy()
    # num_peaks and rel_threshold are set to limit computational burder when models do random predictions.
    if max_keypoints:
        num_peaks = max_keypoints
    else:
        num_peaks = 20
    keypoints = peak_local_max(
        np_heatmap, min_distance=min_keypoint_pixel_distance, threshold_rel=0.05, num_peaks=num_peaks
    )
    return keypoints[::, ::-1].tolist()  # convert to (u,v) aka (col,row) coord frame from (row,col)


def overlay_image_with_heatmap(img: torch.Tensor, heatmap: torch.Tensor, alpha=0.5) -> Image:
    """
    Overlays image with the predicted heatmap, which is projected from grayscale to the red channel.
    """
    # Create heatmap image in red channel
    viridis = cm.get_cmap("viridis")
    heatmap = viridis(heatmap[0].numpy())
    heatmap = TF.to_tensor(heatmap[:, :, 0:3])
    heatmap = heatmap.type(torch.float32)
    img = img.detach().cpu()
    overlay = alpha * img + (1 - alpha) * heatmap
    overlay = TF.to_pil_image(overlay.cpu())

    return overlay
