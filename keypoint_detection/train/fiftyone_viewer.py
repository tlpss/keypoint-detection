"""use fiftyone to visualize the predictions of trained keypoint detectors on a dataset. Very useful for debugging and understanding the models predictions."""
import os

import fiftyone as fo
import torch
import tqdm

from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.train.utils import parse_channel_configuration
from keypoint_detection.utils.heatmap import compute_keypoint_probability, get_keypoints_from_heatmap
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint


def visualize_predictions(
    dataset_path: str,
    models: dict[str, KeypointDetector],
    channel_config: str,
    detect_only_visible_keypoints: bool = False,
):
    """visualize keypoint detectors on a coco dataset. Requires the  coco json, thechannel config and a dict of wandb checkpoints."""

    parsed_channel_config = parse_channel_configuration(channel_config)
    dataset = COCOKeypointsDataset(
        dataset_path, parsed_channel_config, detect_only_visible_keypoints=detect_only_visible_keypoints
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # set all models to eval mode to be sure.
    for model in models.values():
        model.eval()

    predictions = {model_name: [] for model_name in models.keys()}
    gt = []

    # iterate over dataset and compute predictions & gt keypoints
    with torch.no_grad():
        for image, keypoints in tqdm.tqdm(dataloader):
            # [[channel1], [[[x,y],[x,y]]]
            gt_keypoints = []
            for channel in keypoints:
                gt_keypoints.append([[kp[0].item(), kp[1].item()] for kp in channel])
            gt.append(gt_keypoints)

            for model_name, model in models.items():
                heatmaps = model(image)[0]
                # extract keypoints from heatmaps for each channel
                predicted_keypoints = [get_keypoints_from_heatmap(heatmap, 3) for heatmap in heatmaps]
                predicted_keypoint_probabilities = [
                    compute_keypoint_probability(heatmaps[i], predicted_keypoints[i]) for i in range(len(heatmaps))
                ]
                predictions[model_name].append([predicted_keypoints, predicted_keypoint_probabilities])

    ## create fiftyone dataset and add the predictions and gt

    fo_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.dirname(dataset_path),
        label_types=None,
        labels_path=dataset_path,
    )

    fo_dataset.add_dynamic_sample_fields()

    # add the ground truth to the dataset
    for sample_idx, sample in enumerate(fo_dataset):
        _add_instance_keypoints_to_fo_sample(
            sample, "ground_truth_keypoints", gt[sample_idx], None, parsed_channel_config
        )

    # add the predictions to the dataset
    for model_name, model in models.items():
        for sample_idx, sample in enumerate(fo_dataset):
            keypoints, probabilities = predictions[model_name][sample_idx]
            _add_instance_keypoints_to_fo_sample(
                sample, f"{model_name}_keypoints", keypoints, probabilities, parsed_channel_config
            )

    # could do only one loop instead of two for the predictions usually, but we have to compute the GT keypoints, so we need to loop over the dataset anyway
    # https://docs.voxel51.com/user_guide/dataset_creation/index.html#model-predictions

    print(fo_dataset)
    session = fo.launch_app(dataset=fo_dataset)
    session.wait()


def _add_instance_keypoints_to_fo_sample(
    sample, predictions_name, instance_keypoints, keypoint_probabilities, parsed_channels
):
    """adds the detected keypoints to the sample in the fiftyone format"""
    assert len(instance_keypoints) == len(parsed_channels)
    # assert instance_keypoints[0][0][0] > 1.0 # check if the keypoints are not normalized yet
    fo_keypoints = []
    for channel_idx in range(len(instance_keypoints)):
        channel_keypoints = instance_keypoints[channel_idx]
        # normalize the keypoints to the image size
        width = sample["metadata"]["width"]
        height = sample["metadata"]["height"]
        channel_keypoints = [[kp[0] / width, kp[1] / height] for kp in channel_keypoints]
        if keypoint_probabilities is not None:
            channel_keypoint_probabilities = keypoint_probabilities[channel_idx]
        else:
            channel_keypoint_probabilities = None
        fo_keypoints.append(
            fo.Keypoint(
                label="=".join(parsed_channels[channel_idx]),
                points=channel_keypoints,
                confidence=channel_keypoint_probabilities,
            )
        )

    sample[predictions_name] = fo.Keypoints(keypoints=fo_keypoints)
    sample.save()
    return sample


if __name__ == "__main__":
    checkpoint_dict = {
        "pyflex": "tlips/synthetic-cloth-keypoints/model-oz195ppw:v2",
        # "real-data": "tlips/synthetic-cloth-keypoints/model-0l8jfuk6:v5"
    }

    dataset_path = "/home/tlips/Code/RTFClothes/512x256/towels-test_resized_512x256/towels-test.json"
    channel_config = "corner0=corner1=corner2=corner3"
    detect_only_visible_keypoints = False

    models = {key: get_model_from_wandb_checkpoint(value) for key, value in checkpoint_dict.items()}
    visualize_predictions(dataset_path, models, channel_config, detect_only_visible_keypoints)
