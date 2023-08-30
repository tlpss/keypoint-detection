"""use fiftyone to visualize the predictions of trained keypoint detectors on a dataset. Very useful for debugging and understanding the models predictions."""
import os
from collections import defaultdict
from typing import List, Optional, Tuple

import fiftyone as fo
import numpy as np
import torch
import tqdm

from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.metrics import DetectedKeypoint, Keypoint, KeypointAPMetrics
from keypoint_detection.train.utils import parse_channel_configuration
from keypoint_detection.utils.heatmap import compute_keypoint_probability, get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint


class KeypointModelViewer:
    def __init__(
        self,
        dataset_path: str,
        models: dict[str, KeypointDetector],
        channel_config: str,
        detect_only_visible_keypoints: bool = False,
        n_samples: Optional[int] = None,
    ):
        self.dataset_path = dataset_path
        self.models = models
        self.channel_config = channel_config
        self.detect_only_visible_keypoints = detect_only_visible_keypoints
        self.n_samples = n_samples
        self.parsed_channel_config = parse_channel_configuration(channel_config)

        dataset = COCOKeypointsDataset(
            dataset_path, self.parsed_channel_config, detect_only_visible_keypoints=detect_only_visible_keypoints
        )
        if self.n_samples is not None:
            dataset = torch.utils.data.Subset(dataset, range(0, self.n_samples))

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # create the AP metrics
        self.ap_metrics = {
            name: [KeypointAPMetrics(model.maximal_gt_keypoint_pixel_distances) for _ in self.parsed_channel_config]
            for name, model in models.items()
        }

        # set all models to eval mode to be sure.
        for model in self.models.values():
            model.eval()

        self.predicted_keypoints = {model_name: [] for model_name in models.keys()}
        self.gt_keypoints = []
        # {model: {sample_idx: {channel_idx: [ap_score]}}
        self.ap_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def predict_and_compute_metrics(self):
        with torch.no_grad():
            sample_idx = 0
            for image, keypoints in tqdm.tqdm(self.dataloader):
                # [[channel1], [[[x,y],[x,y]]]
                gt_keypoints = []
                for channel in keypoints:
                    gt_keypoints.append([[kp[0].item(), kp[1].item()] for kp in channel])
                self.gt_keypoints.append(gt_keypoints)

                for model_name, model in self.models.items():
                    heatmaps = model(image)[0]
                    # extract keypoints from heatmaps for each channel
                    predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps.unsqueeze(0))[0]
                    predicted_keypoint_probabilities = [
                        compute_keypoint_probability(heatmaps[i], predicted_keypoints[i]) for i in range(len(heatmaps))
                    ]
                    self.predicted_keypoints[model_name].append(
                        [predicted_keypoints, predicted_keypoint_probabilities]
                    )

                    #### METRIC COMPUTATION ####
                    for metric in self.ap_metrics[model_name]:
                        metric.reset()

                    for channel_idx in range(len(self.parsed_channel_config)):
                        metric_detected_keypoints = predicted_keypoints[channel_idx]
                        probabilities = predicted_keypoint_probabilities[channel_idx]
                        metric_detected_keypoints = [
                            DetectedKeypoint(kp[0], kp[1], p)
                            for kp, p in zip(metric_detected_keypoints, probabilities)
                        ]
                        metric_gt_formatted_keypoints = [Keypoint(kp[0], kp[1]) for kp in gt_keypoints[channel_idx]]
                        self.ap_metrics[model_name][channel_idx].update(
                            metric_detected_keypoints, metric_gt_formatted_keypoints
                        )

                    for channel_idx in range(len(self.parsed_channel_config)):
                        self.ap_scores[model_name][sample_idx].update(
                            {channel_idx: list(self.ap_metrics[model_name][channel_idx].compute().values())}
                        )

                sample_idx += 1

    def visualize_predictions(
        self,
    ):
        """visualize keypoint detectors on a coco dataset. Requires the  coco json, thechannel config and a dict of wandb checkpoints."""

        # iterate over dataset and compute predictions & gt keypoints as well as metrics

        ## create fiftyone dataset and add the predictions and gt

        fo_dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=os.path.dirname(self.dataset_path),
            label_types=[],  # do not load the coco annotations
            labels_path=self.dataset_path,
        )

        fo_dataset.add_dynamic_sample_fields()

        fo_dataset = fo_dataset.limit(self.n_samples)

        # add the ground truth to the dataset
        for sample_idx, sample in enumerate(fo_dataset):
            self._add_instance_keypoints_to_fo_sample(
                sample, "ground_truth_keypoints", self.gt_keypoints[sample_idx], None, self.parsed_channel_config
            )

        # add the predictions to the dataset
        for model_name, model in self.models.items():
            for sample_idx, sample in enumerate(fo_dataset):
                keypoints, probabilities = self.predicted_keypoints[model_name][sample_idx]
                self._add_instance_keypoints_to_fo_sample(
                    sample, f"{model_name}_keypoints", keypoints, probabilities, self.parsed_channel_config
                )
                model_ap_scores = self.ap_scores[model_name][sample_idx]

                # log map
                ap_values = np.zeros((len(self.parsed_channel_config), len(model.maximal_gt_keypoint_pixel_distances)))
                for channel_idx in range(len(self.parsed_channel_config)):
                    for max_dist_idx in range(len(model.maximal_gt_keypoint_pixel_distances)):
                        ap_values[channel_idx, max_dist_idx] = model_ap_scores[channel_idx][max_dist_idx]
                sample[f"{model_name}_keypoints_mAP"] = ap_values.mean()
                sample.save()
        # could do only one loop instead of two for the predictions usually, but we have to compute the GT keypoints, so we need to loop over the dataset anyway
        # https://docs.voxel51.com/user_guide/dataset_creation/index.html#model-predictions

        print(fo_dataset)

        session = fo.launch_app(dataset=fo_dataset)
        session = self._configure_session_colors(session)
        session.wait()

    def _configure_session_colors(self, session: fo.Session) -> fo.Session:
        """
        set colors such that each model has a different color and the mAP labels have the same color as the keypoints.
        """

        # chatgpt color pool
        color_pool = [
            "#FF00FF",  # Neon Purple
            "#00FF00",  # Electric Green
            "#FFFF00",  # Cyber Yellow
            "#0000FF",  # Laser Blue
            "#FF0000",  # Radioactive Red
            "#00FFFF",  # Galactic Teal
            "#FF00AA",  # Quantum Pink
            "#C0C0C0",  # Holographic Silver
            "#000000",  # Abyssal Black
            "#FFA500",  # Cosmic Orange
        ]
        color_fields = []
        color_fields.append({"path": "ground_truth_keypoints", "fieldColor": color_pool[-1]})
        for model_idx, model_name in enumerate(self.models.keys()):
            color_fields.append({"path": f"{model_name}_keypoints", "fieldColor": color_pool[model_idx]})
            color_fields.append({"path": f"{model_name}_keypoints_mAP", "fieldColor": color_pool[model_idx]})
        session.color_scheme = fo.ColorScheme(color_pool=color_pool, fields=color_fields)
        return session

    def _add_instance_keypoints_to_fo_sample(
        self,
        sample,
        predictions_name,
        instance_keypoints: List[List[Tuple]],
        keypoint_probabilities: List[List[float]],
        parsed_channels: List[List[str]],
    ) -> fo.Sample:
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
    # TODO: make CLI for this -> hydra config?
    checkpoint_dict = {
        "maxvit-256-flat": "tlips/synthetic-cloth-keypoints-quest-for-precision/model-5ogj44k0:v0",
        "maxvit-512-flat": "tlips/synthetic-cloth-keypoints-quest-for-precision/model-1of5e6qs:v0",
    }

    dataset_path = "/storage/users/tlips/RTFClothes/towels-test_resized_256x256/towels-test.json"
    channel_config = "corner0=corner1=corner2=corner3"
    detect_only_visible_keypoints = False
    n_samples = 100
    models = {key: get_model_from_wandb_checkpoint(value) for key, value in checkpoint_dict.items()}
    visualizer = KeypointModelViewer(dataset_path, models, channel_config, detect_only_visible_keypoints, n_samples)
    visualizer.predict_and_compute_metrics()
    visualizer.visualize_predictions()
