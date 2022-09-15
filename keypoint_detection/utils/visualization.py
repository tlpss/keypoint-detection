from typing import List

import pytorch_lightning.loggers
import torch
import torchvision
import wandb
from matplotlib import cm

from keypoint_detection.utils.heatmap import generate_channel_heatmap, get_keypoints_from_heatmap


def overlay_image_with_heatmap(images: torch.Tensor, heatmaps: torch.Tensor, alpha=0.5) -> torch.Tensor:
    """ """
    viridis = cm.get_cmap("viridis")
    heatmaps = viridis(heatmaps.numpy())[..., :3]  # viridis: grayscale -> RGBa
    heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
    heatmaps = heatmaps.permute((0, 3, 1, 2))  # HxWxC -> CxHxW for pytorch

    overlayed_images = alpha * images + (1 - alpha) * heatmaps
    return overlayed_images


def overlay_image_with_keypoints(images: torch.Tensor, keypoints: List[torch.Tensor], sigma: float) -> torch.Tensor:
    """
    images N x 3 x H x W
    keypoints list of size N with Tensors C x 2


    Returns:
        torch.Tensor: N x 3 x H x W
    """

    image_size = images.shape[2:]
    alpha = 0.7
    keypoint_color = torch.Tensor([240.0, 240.0, 10.0]) / 255.0
    keypoint_color = keypoint_color.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    overlayed_images = []
    for i in range(images.shape[0]):

        heatmaps = generate_channel_heatmap(image_size, keypoints[i], sigma=sigma, device="cpu")  # C x H x W
        heatmaps = heatmaps.unsqueeze(0)  # 1 xC x H x W
        colorized_heatmaps = keypoint_color * heatmaps
        combined_heatmap = torch.max(colorized_heatmaps, dim=1)[0]  # 3 x H x W
        combined_heatmap[combined_heatmap < 0.1] = 0.0  # avoid glare

        overlayed_image = images[i] * alpha + combined_heatmap
        overlayed_image = torch.clip(overlayed_image, 0.0, 1.0)
        overlayed_images.append(overlayed_image)
    overlayed_images = torch.stack(overlayed_images)
    return overlayed_images


def visualize_predictions(
    imgs: torch.Tensor,
    predicted_heatmaps: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    logger: pytorch_lightning.loggers.WandbLogger,
    minimal_keypoint_pixel_distance: int,
    keypoint_channel: str,
    is_validation_step: bool = True,
):
    num_images = min(predicted_heatmaps.shape[0], 6)
    keypoint_sigma = max(1, imgs.shape[2] / 64)

    predicted_heatmap_overlays = overlay_image_with_heatmap(imgs[:num_images], predicted_heatmaps[:num_images])

    gt_heatmap_overlays = overlay_image_with_heatmap(imgs[:num_images], gt_heatmaps[:num_images])

    keypoints = [
        torch.tensor(get_keypoints_from_heatmap(predicted_heatmaps[i].cpu(), minimal_keypoint_pixel_distance))
        for i in range(predicted_heatmaps.shape[0])
    ]
    predicted_keypoints_overlays = overlay_image_with_keypoints(
        imgs[:num_images], keypoints[:num_images], keypoint_sigma
    )

    images = torch.cat([predicted_heatmap_overlays, predicted_keypoints_overlays, gt_heatmap_overlays])

    grid = torchvision.utils.make_grid(images, nrow=num_images)
    mode = "val" if is_validation_step else "train"

    if isinstance(keypoint_channel, list):
        if len(keypoint_channel) == 1:
            keypoint_channel = keypoint_channel[0]
        else:
            keypoint_channel = f"{keypoint_channel[0]}+{keypoint_channel[1]}+..."
    keypoint_channel_short = (keypoint_channel[:40] + "...") if len(keypoint_channel) > 40 else keypoint_channel
    label = f"{keypoint_channel_short}_{mode}_keypoints"
    logger.experiment.log(
        {label: wandb.Image(grid, caption="top: predicted heatmaps, middle: predicted keypoints, bottom: gt heatmap")}
    )
