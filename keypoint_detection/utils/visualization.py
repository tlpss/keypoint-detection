from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

from keypoint_detection.utils.heatmap import generate_channel_heatmap

# Kelly's 22 colors for max contrast
# cf. https://gist.github.com/ollieglass/f6ddd781eeae1d24e391265432297538
DISTINCT_COLORS = [
    "#F2F3F4",
    "#222222",
    "#F3C300",
    "#875692",
    "#F38400",
    "#A1CAF1",
    "#BE0032",
    "#C2B280",
    "#848482",
    "#008856",
    "#E68FAC",
    "#0067A5",
    "#F99379",
    "#604E97",
    "#F6A600",
    "#B3446C",
    "#DCD300",
    "#882D17",
    "#8DB600",
    "#654522",
    "#E25822",
    "#2B3D26",
]


def get_logging_label_from_channel_configuration(channel_configuration: List[List[str]], mode: str) -> str:
    channel_name = channel_configuration

    if isinstance(channel_configuration, list):
        if len(channel_configuration) == 1:
            channel_name = channel_configuration[0]
        else:
            channel_name = f"{channel_configuration[0]}+{channel_configuration[1]}+..."

    channel_name_short = (channel_name[:40] + "...") if len(channel_name) > 40 else channel_name
    if mode != "":
        label = f"{channel_name_short}_{mode}"
    else:
        label = channel_name_short
    return label


def overlay_image_with_heatmap(images: torch.Tensor, heatmaps: torch.Tensor, alpha=0.5) -> torch.Tensor:
    """ """
    viridis = cm.get_cmap("viridis")
    heatmaps = viridis(heatmaps.numpy())[..., :3]  # viridis: grayscale -> RGBa
    heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
    heatmaps = heatmaps.permute((0, 3, 1, 2))  # HxWxC -> CxHxW for pytorch

    overlayed_images = alpha * images + (1 - alpha) * heatmaps
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


def overlay_images_with_keypoints(images: torch.Tensor, keypoints: List[torch.Tensor], sigma: float) -> torch.Tensor:
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


def draw_keypoints_on_image(
    image: Image, image_keypoints: List[List[Tuple[int, int]]], channel_configuration: List[List[str]]
) -> Image:
    """adds all keypoints to the PIL image, with different colors for each channel."""
    color_pool = DISTINCT_COLORS
    image_size = image.size
    min_size = min(image_size)
    scale = 1 + (min_size // 256)

    draw = ImageDraw.Draw(image)
    for channel_idx, channel_keypoints in enumerate(image_keypoints):
        for keypoint_idx, keypoint in enumerate(channel_keypoints):
            u, v = keypoint
            draw.ellipse((u - scale, v - scale, u + scale, v + scale), fill=color_pool[channel_idx])

        draw.text(
            (10, channel_idx * 10 * scale),
            get_logging_label_from_channel_configuration(channel_configuration[channel_idx], ""),
            fill=color_pool[channel_idx],
            font=ImageFont.truetype("FreeMono.ttf", size=10 * scale),
        )

    return image


def visualize_predicted_keypoints(
    images: torch.Tensor, keypoints: List[List[List[List[int]]]], channel_configuration: List[List[str]]
):
    drawn_images = []
    num_images = min(images.shape[0], 6)
    for i in range(num_images):
        # PIL expects uint8 images
        image = images[i].permute(1, 2, 0).numpy() * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image = draw_keypoints_on_image(image, keypoints[i], channel_configuration)
        drawn_images.append(image)

    drawn_images = torch.stack([torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255 for image in drawn_images])

    grid = torchvision.utils.make_grid(drawn_images, nrow=num_images)
    return grid


if __name__ == "__main__":
    """Script to visualize dataset"""
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
    from keypoint_detection.tasks.train import parse_channel_configuration
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
