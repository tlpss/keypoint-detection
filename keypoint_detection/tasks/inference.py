""" run inference on a provided image and save the result to a file """

import numpy as np
import torch
from PIL import Image

from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from keypoint_detection.utils.visualization import draw_keypoints_on_image


def run_inference(model: KeypointDetector, image, confidence_threshold: float = 0.1) -> Image:
    model.eval()
    tensored_image = torch.from_numpy(np.array(image)).float()
    tensored_image = tensored_image / 255.0
    tensored_image = tensored_image.permute(2, 0, 1)
    tensored_image = tensored_image.unsqueeze(0)
    with torch.no_grad():
        heatmaps = model(tensored_image)

    keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, abs_max_threshold=confidence_threshold)
    image_keypoints = keypoints[0]
    for keypoints, channel_config in zip(image_keypoints, model.keypoint_channel_configuration):
        print(f"Keypoints for {channel_config}: {keypoints}")
    image = draw_keypoints_on_image(image, image_keypoints, model.keypoint_channel_configuration)
    return image


if __name__ == "__main__":
    wandb_checkpoint = "tlips/synthetic-lego-battery-keypoints/model-tbzd50z8:v0"
    image_path = "/home/tlips/Downloads/Lego-battery-real/0.jpg"
    # image_path = "/home/tlips/Documents/synthetic-cloth-data/synthetic-cloth-data/data/datasets/LEGO-battery/01/images/0.jpg"
    image_size = (256, 256)

    image = Image.open(image_path)
    image = image.resize(image_size)

    model = get_model_from_wandb_checkpoint(wandb_checkpoint)
    image = run_inference(model, image)
    image.save("inference_result.png")
