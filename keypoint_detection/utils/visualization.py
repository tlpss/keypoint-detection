import pytorch_lightning.loggers
import torch
import torchvision

import wandb
from keypoint_detection.utils.heatmap import (
    generate_keypoints_heatmap,
    get_keypoints_from_heatmap,
    overlay_image_with_heatmap,
)


def visualize_predictions(
    imgs: torch.Tensor,
    predicted_heatmaps: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    logger: pytorch_lightning.loggers.WandbLogger,
    minimal_keypoint_pixel_distance: int,
    keypoint_channel: str,
    validate: bool = True,
):
    num_images = min(predicted_heatmaps.shape[0], 6)
    transform = torchvision.transforms.ToTensor()

    # corners
    overlayed_predicted_heatmap = torch.stack(
        [
            transform(overlay_image_with_heatmap(imgs[i], torch.unsqueeze(predicted_heatmaps[i].cpu(), 0)))
            for i in range(num_images)
        ]
    )
    overlayed_gt = torch.stack(
        [
            transform(overlay_image_with_heatmap(imgs[i], torch.unsqueeze(gt_heatmaps[i].cpu(), 0)))
            for i in range(num_images)
        ]
    )

    overlayed_predicted_keypoints = torch.stack(
        [
            transform(
                overlay_image_with_heatmap(
                    imgs[i],
                    torch.unsqueeze(
                        generate_keypoints_heatmap(
                            imgs.shape[-2:],
                            get_keypoints_from_heatmap(predicted_heatmaps[i].cpu(), minimal_keypoint_pixel_distance),
                            sigma=max(1, int(imgs.shape[-1] / 64)),
                            device="cpu",
                        ),
                        0,
                    ),
                )
            )
            for i in range(num_images)
        ]
    )
    images = torch.cat([overlayed_predicted_heatmap, overlayed_predicted_keypoints, overlayed_gt])

    grid = torchvision.utils.make_grid(images, nrow=num_images)
    mode = "val" if validate else "train"
    label = f"{keypoint_channel}_{mode}_keypoints"
    logger.experiment.log(
        {label: wandb.Image(grid, caption="top: predicted heatmaps, middle: predicted keypoints, bottom: gt heatmap")}
    )
