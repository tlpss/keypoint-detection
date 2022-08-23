import pytorch_lightning.loggers
import torch
import torchvision
import wandb
from matplotlib import cm
from PIL import Image

from keypoint_detection.utils.heatmap import generate_channel_heatmap, get_keypoints_from_heatmap


def overlay_image_with_heatmap(images: torch.Tensor, heatmaps: torch.Tensor, alpha=0.5) -> Image:
    """ """
    # Create heatmap image in red channel
    viridis = cm.get_cmap("viridis")
    heatmaps = viridis(heatmaps.numpy())[..., :3]  # viridis: grayscale -> RGBa
    heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
    heatmaps = heatmaps.permute((0, 3, 1, 2))  # HxWxC -> CxHxW for pytorch

    overlayed_images = alpha * images + (1 - alpha) * heatmaps
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

    predicted_heatmap_overlays = overlay_image_with_heatmap(imgs, predicted_heatmaps)
    gt_heatmap_overlays = overlay_image_with_heatmap(imgs, gt_heatmaps)

    predicted_keypoints_overlays = overlay_image_with_heatmap(
        imgs,
        torch.stack(
            [
                generate_channel_heatmap(
                    imgs.shape[-2:],
                    torch.tensor(
                        get_keypoints_from_heatmap(predicted_heatmaps[i].cpu(), minimal_keypoint_pixel_distance)
                    ),
                    sigma=max(1, int(imgs.shape[-1] / 64)),
                    device="cpu",
                )
                for i in range(predicted_heatmaps.shape[0])
            ]
        ),
    )

    images = torch.cat([predicted_heatmap_overlays, predicted_keypoints_overlays, gt_heatmap_overlays])

    grid = torchvision.utils.make_grid(images, nrow=num_images)
    mode = "val" if is_validation_step else "train"
    label = f"{keypoint_channel}_{mode}_keypoints"
    logger.experiment.log(
        {label: wandb.Image(grid, caption="top: predicted heatmaps, middle: predicted keypoints, bottom: gt heatmap")}
    )
