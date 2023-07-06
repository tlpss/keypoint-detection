import os
from typing import List

import fiftyone as fo
import torch
import tqdm

from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.train.utils import parse_channel_configuration
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint


def visualize_predictions(
    dataset_path: str, models: List[KeypointDetector], channel_config: str, detect_only_visible_keypoints: bool = False
):
    parsed_channel_config = parse_channel_configuration(channel_config)
    dataset = COCOKeypointsDataset(
        dataset_path, parsed_channel_config, detect_only_visible_keypoints=detect_only_visible_keypoints
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # set all models to eval mode to be sure.
    for model in models:
        model.eval()

    predictions = [[] for _ in range(len(models) + 1)]
    gt = []
    ["ground_truth"] + [f"model_{i}" for i in range(len(models))]
    with torch.no_grad():
        for image, keypoints in tqdm.tqdm(dataloader):
            keypoints = [[kp[0].item(), kp[1].item()] for channel in keypoints for kp in channel]
            # [[channel1], [[[x,y],[x,y]]]
            gt.append(keypoints)

            for model_idx, model in enumerate(models):
                heatmaps = model(image)[0]
                # extract keypoints from heatmaps for each channel
                predicted_keypoints = [get_keypoints_from_heatmap(heatmap, 3) for heatmap in heatmaps]
                predictions[model_idx].append(predicted_keypoints)

    fo_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.dirname(dataset_path),
        label_types=["keypoints"],
        labels_path=dataset_path,
    )

    fo_dataset.add_dynamic_sample_fields()

    # add the predictions to the dataset
    for model_idx in range(len(models)):
        for sample_idx, sample in enumerate(fo_dataset):
            keypoints = predictions[model_idx][sample_idx]
            fo_keypoints = []
            for channel_idx in range(len(keypoints)):
                channel_keypoints = keypoints[channel_idx]
                # normalize the keypoints to the image size
                width = sample["metadata"]["width"]
                height = sample["metadata"]["height"]
                channel_keypoints = [[kp[0] / width, kp[1] / height] for kp in channel_keypoints]

                fo_keypoints.append(fo.Keypoint(label="test", points=channel_keypoints))

                sample[f"predictions_model_{model_idx}"] = fo.Keypoints(keypoints=fo_keypoints)

            sample.save()
            # fo_dataset.add_sample(sample)

    print(fo_dataset)
    session = fo.launch_app(dataset=fo_dataset)
    session.wait()

    # TODO: cleanup
    # TODO: more descriptive model names?
    # TODO: add channel names instead of 'test'
    # TODO: only one loop instead of two for the predictions? (but also have to obtain GT predictions ofc.)
    # https://docs.voxel51.com/user_guide/dataset_creation/index.html#model-predictions


if __name__ == "__main__":
    checkpoints = [
        "tlips/synthetic-cloth-keypoints/model-0l8jfuk6:v5",
        "tlips/synthetic-cloth-keypoints/model-0l8jfuk6:v5",
    ]
    dataset_path = "/home/tlips/Code/RTFClothes/512x256/towels-train_resized_512x256/towels-train.json"
    channel_config = "corner0=corner1=corner2=corner3"
    detect_only_visible_keypoints = False

    models = [get_model_from_wandb_checkpoint(checkpoint) for checkpoint in checkpoints]
    visualize_predictions(dataset_path, models, channel_config, detect_only_visible_keypoints)
