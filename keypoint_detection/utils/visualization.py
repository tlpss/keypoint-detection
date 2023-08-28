from argparse import ArgumentParser
from typing import List

import torch
import torchvision
from matplotlib import cm

from keypoint_detection.utils.heatmap import generate_channel_heatmap


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


def visualize_predicted_heatmaps(
    imgs: torch.Tensor,
    predicted_heatmaps: torch.Tensor,
    gt_heatmaps: torch.Tensor,
):
    num_images = min(predicted_heatmaps.shape[0], 6)

    predicted_heatmap_overlays = overlay_image_with_heatmap(imgs[:num_images], predicted_heatmaps[:num_images])
    gt_heatmap_overlays = overlay_image_with_heatmap(imgs[:num_images], gt_heatmaps[:num_images])

    images = torch.cat([predicted_heatmap_overlays, gt_heatmap_overlays])
    grid = torchvision.utils.make_grid(images, nrow=num_images)
    return grid


if __name__ == "__main__":
    """Script to visualize dataset"""
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
    from keypoint_detection.train.train import parse_channel_configuration
    from keypoint_detection.utils.heatmap import create_heatmap_batch

    parser = ArgumentParser()
    parser.add_argument("json_dataset_path")
    parser.add_argument("keypoint_channel_configuration")
    args = parser.parse_args()

    hparams = vars(parser.parse_args())
    hparams["keypoint_channel_configuration"] = parse_channel_configuration(hparams["keypoint_channel_configuration"])

    dataset = COCOKeypointsDataset(**hparams)
    batch_size = 6
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    images, keypoint_channels = next(iter(dataloader))

    shape = images.shape[2:]

    heatmaps = create_heatmap_batch(shape, keypoint_channels[0], sigma=6.0, device="cpu")
    grid = visualize_predicted_heatmaps(images, heatmaps, heatmaps, 6)

    image_numpy = grid.permute(1, 2, 0).numpy()
    plt.imshow(image_numpy)
    plt.show()
