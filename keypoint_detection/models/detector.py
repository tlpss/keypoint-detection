import argparse
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.metrics import DetectedKeypoint, Keypoint, KeypointAPMetrics
from keypoint_detection.utils.heatmap import BCE_loss, create_heatmap_batch, get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.visualization import (
    get_logging_label_from_channel_configuration,
    visualize_predicted_heatmaps,
    visualize_predicted_keypoints,
)


class KeypointDetector(pl.LightningModule):
    """
    keypoint Detector using Spatial Heatmaps.
    There can be N channels of keypoints, each with its own set of ground truth keypoints.
    The mean Average precision is used to calculate the performance.

    """

    @staticmethod
    def add_model_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        """
        parser = parent_parser.add_argument_group("KeypointDetector")

        parser.add_argument(
            "--heatmap_sigma",
            default=2,
            type=int,
            help="The size of the Gaussian blobs that are used to create the ground truth heatmaps from the keypoint coordinates.",
        )
        parser.add_argument(
            "--minimal_keypoint_extraction_pixel_distance",
            type=int,
            default=1,
            help="the minimal pixel-distance between two keypoints. Allows for some non-maximum surpression.",
        )
        parser.add_argument(
            "--maximal_gt_keypoint_pixel_distances",
            type=str,
            default="2 4",
            help="The treshold distance(s) for the AP metric to consider a detection as a True Positive. Separate multiple values by a space to compute the AP for all values.",
        )
        parser.add_argument("--learning_rate", type=float, default=3e-4)  # Karpathy constant
        parser.add_argument(
            "--ap_epoch_start",
            type=int,
            default=1,
            help="Epoch at which to start calculating the AP every `ap_epoch_frequency` epochs.",
        )
        parser.add_argument(
            "--ap_epoch_freq",
            type=int,
            default=2,
            help="Rate at which to calculate the AP metric if epoch > `ap_epoch_start`",
        )
        parser.add_argument(
            "--lr_scheduler_relative_threshold",
            default=0.0,
            type=float,
            help="relative threshold for the OnPlateauLRScheduler. If the training epoch loss does not decrease with this fraction for 2 consective epochs, lr is decreased with factor 10.",
        )
        parser.add_argument(
            "--max_keypoints",
            default=20,
            type=int,
            help="the maximum number of keypoints to predict from the generated heatmaps. If set to -1, skimage will look for all peaks in the heatmap, if set to N (N>0) it will return the N most most certain ones.",
        )
        return parent_parser

    def __init__(
        self,
        heatmap_sigma: int,
        maximal_gt_keypoint_pixel_distances: str,
        minimal_keypoint_extraction_pixel_distance: int,
        learning_rate: float,
        backbone: Backbone,
        keypoint_channel_configuration: List[List[str]],
        ap_epoch_start: int,
        ap_epoch_freq: int,
        lr_scheduler_relative_threshold: float,
        max_keypoints: int,
        **kwargs,
    ):
        """[summary]

        Args:
            see argparse help strings for documentation.

            kwargs: Pythonic catch for the other named arguments, used so that we can use a dict with ALL system hyperparameters to initialise the model from this
                    hyperparamater configuration dict. The alternative is to add a single 'hparams' argument to the init function, but this is imo less readable.
                    cf https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html for an overview.
        """
        super().__init__()
        ## No need to manage devices ourselves, pytorch.lightning does all of that.
        ## device can be accessed through self.device if required.

        # to add new hyperparameters:
        # 1. define as named arg in the init (and use them)
        # 2. add to the argparse method of this module
        # 3. pass them along when calling the train.py file to override their default value

        self.learning_rate = learning_rate
        self.heatmap_sigma = heatmap_sigma
        self.ap_epoch_start = ap_epoch_start
        self.ap_epoch_freq = ap_epoch_freq
        self.minimal_keypoint_pixel_distance = minimal_keypoint_extraction_pixel_distance
        self.lr_scheduler_relative_threshold = lr_scheduler_relative_threshold
        self.max_keypoints = max_keypoints
        self.keypoint_channel_configuration = keypoint_channel_configuration
        # parse the gt pixel distances
        if isinstance(maximal_gt_keypoint_pixel_distances, str):
            maximal_gt_keypoint_pixel_distances = [
                int(val) for val in maximal_gt_keypoint_pixel_distances.strip().split(" ")
            ]
        self.maximal_gt_keypoint_pixel_distances = maximal_gt_keypoint_pixel_distances

        self.ap_training_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channel_configuration
        ]
        self.ap_validation_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channel_configuration
        ]

        self.ap_test_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channel_configuration
        ]

        self.n_heatmaps = len(self.keypoint_channel_configuration)

        head = nn.Conv2d(
            in_channels=backbone.get_n_channels_out(),
            out_channels=self.n_heatmaps,
            kernel_size=(3, 3),
            padding="same",
        )

        # expect output of backbone to be normalized!
        # so by filling bias to -4, the sigmoid should be on avg sigmoid(-4) =  0.02
        # which is consistent with the desired heatmaps that are zero almost everywhere.
        # setting too low would result in loss of gradients..
        head.bias.data.fill_(-4)

        self.unnormalized_model = nn.Sequential(
            backbone,
            head,
        )  # NO sigmoid to combine it in the loss! (needed for FP16)

        # save hyperparameters to logger, to make sure the model hparams are saved even if
        # they are not included in the config (i.e. if they are kept at the defaults).
        # this is for later reference (e.g. checkpoint loading) and consistency.
        self.save_hyperparameters(ignore=["**kwargs", "backbone"])

    def forward(self, x: torch.Tensor):
        """
        x shape must be of shape (N,3,H,W)
        returns tensor with shape (N, n_heatmaps, H,W)
        """
        return torch.sigmoid(self.forward_unnormalized(x))

    def forward_unnormalized(self, x: torch.Tensor):
        return self.unnormalized_model(x)

    def configure_optimizers(self):
        """
        Configures an Adam optimizer.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)

        # TODO: the LR scheduler can reduce training robustness for smaller datasets (where the val loss might fluctuate more),
        # this makes the impact of a random seed larger. So for now, we disable it.
        # solution would be to limit the dependency of the scheduler on the dataset size, e.g. by coupling it to the number of model updates instead of epochs.

        # self.lr_scheduler = ReduceLROnPlateau(
        #     self.optimizer,
        #     threshold=self.lr_scheduler_relative_threshold,
        #     threshold_mode="rel",
        #     mode="min",
        #     factor=0.1,
        #     patience=2,
        #     verbose=True,
        # )
        return {
            "optimizer": self.optimizer,
        }

    def shared_step(self, batch, batch_idx, include_visualization_data_in_result_dict=False) -> Dict[str, Any]:
        """
        shared step for train and validation step that computes the heatmaps and losses and
        creates a result dict for later use in the train, validate and test step.

        batch: img, keypoints
        where img is a Nx3xHxW tensor
        and keypoints a nested list of len(channels) x N with K_ij x 2 tensors containing the keypoints for each channel and each sample in the batch

        returns:

        shared_dict (Dict): a dict with a.o. heatmaps, gt_keypoints and losses
        """
        input_images, keypoint_channels = batch
        heatmap_shape = input_images[0].shape[1:]

        gt_heatmaps = [
            create_heatmap_batch(heatmap_shape, keypoint_channel, self.heatmap_sigma, self.device)
            for keypoint_channel in keypoint_channels
        ]

        input_images = input_images.to(self.device)

        ## predict and compute losses
        predicted_unnormalized_maps = self.forward_unnormalized(input_images)
        predicted_heatmaps = torch.sigmoid(predicted_unnormalized_maps)
        channel_losses = []
        channel_gt_losses = []

        result_dict = {}
        for channel_idx in range(len(self.keypoint_channel_configuration)):
            channel_losses.append(
                # combines sigmoid with BCE for increased stability.
                nn.functional.binary_cross_entropy_with_logits(
                    predicted_unnormalized_maps[:, channel_idx, :, :], gt_heatmaps[channel_idx]
                )
            )
            with torch.no_grad():
                channel_gt_losses.append(BCE_loss(gt_heatmaps[channel_idx], gt_heatmaps[channel_idx]))

            # pass losses and other info to result dict
            result_dict.update(
                {f"{self.keypoint_channel_configuration[channel_idx]}_loss": channel_losses[channel_idx].detach()}
            )

        loss = sum(channel_losses)
        gt_loss = sum(channel_gt_losses)
        result_dict.update({"loss": loss, "gt_loss": gt_loss})

        if include_visualization_data_in_result_dict:
            result_dict.update(
                {
                    "input_images": input_images.detach().cpu(),
                    "gt_keypoints": keypoint_channels,
                    "predicted_heatmaps": predicted_heatmaps.detach().cpu(),
                    "gt_heatmaps": gt_heatmaps,
                }
            )

        return result_dict

    def training_step(self, train_batch, batch_idx):
        log_images = batch_idx == 0 and self.current_epoch > 0 and self.is_ap_epoch()
        should_log_ap = self.is_ap_epoch() and batch_idx < 20  # limit AP calculation to first 20 batches to save time
        include_vis_data = log_images or should_log_ap

        result_dict = self.shared_step(
            train_batch, batch_idx, include_visualization_data_in_result_dict=include_vis_data
        )

        if should_log_ap:
            self.update_ap_metrics(result_dict, self.ap_training_metrics)

        if log_images:
            image_grids = self.visualize_predictions_channels(result_dict)
            self.log_channel_predictions_grids(image_grids, mode="train")

        for channel_name in self.keypoint_channel_configuration:
            self.log(f"train/{channel_name}", result_dict[f"{channel_name}_loss"])

        # self.log("train/loss", result_dict["loss"])
        self.log("train/gt_loss", result_dict["gt_loss"])
        self.log("train/loss", result_dict["loss"], on_epoch=True)  # also logs steps?
        return result_dict

    def update_ap_metrics(self, result_dict, ap_metrics):
        predicted_heatmaps = result_dict["predicted_heatmaps"]
        gt_keypoints = result_dict["gt_keypoints"]
        for channel_idx in range(len(self.keypoint_channel_configuration)):
            predicted_heatmaps_channel = predicted_heatmaps[:, channel_idx, :, :]
            gt_keypoints_channel = gt_keypoints[channel_idx]
            self.update_channel_ap_metrics(predicted_heatmaps_channel, gt_keypoints_channel, ap_metrics[channel_idx])

    def visualize_predictions_channels(self, result_dict):
        input_images = result_dict["input_images"]
        gt_heatmaps = result_dict["gt_heatmaps"]
        predicted_heatmaps = result_dict["predicted_heatmaps"]

        image_grids = []
        for channel_idx in range(len(self.keypoint_channel_configuration)):
            grid = visualize_predicted_heatmaps(
                input_images,
                predicted_heatmaps[:, channel_idx, :, :],
                gt_heatmaps[channel_idx].cpu(),
            )
            image_grids.append(grid)
        return image_grids

    def log_channel_predictions_grids(self, image_grids, mode: str):
        for channel_configuration, grid in zip(self.keypoint_channel_configuration, image_grids):
            label = get_logging_label_from_channel_configuration(channel_configuration, mode)
            image_caption = "top: predicted heatmaps, bottom: gt heatmaps"
            self.logger.experiment.log({label: wandb.Image(grid, caption=image_caption)})

    def visualize_predicted_keypoints(self, result_dict):
        images = result_dict["input_images"]
        predicted_heatmaps = result_dict["predicted_heatmaps"]
        # get the keypoints from the heatmaps
        predicted_heatmaps = predicted_heatmaps.detach().float()
        predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(
            predicted_heatmaps, self.max_keypoints, self.minimal_keypoint_pixel_distance, abs_max_threshold=0.1
        )
        # overlay the images with the keypoints
        grid = visualize_predicted_keypoints(images, predicted_keypoints, self.keypoint_channel_configuration)
        return grid

    def log_predicted_keypoints(self, grid, mode=str):
        label = f"predicted_keypoints_{mode}"
        image_caption = "predicted keypoints"
        self.logger.experiment.log({label: wandb.Image(grid, caption=image_caption)})

    def validation_step(self, val_batch, batch_idx):
        # no need to switch model to eval mode, this is handled by pytorch lightning
        result_dict = self.shared_step(val_batch, batch_idx, include_visualization_data_in_result_dict=True)

        if self.is_ap_epoch():
            self.update_ap_metrics(result_dict, self.ap_validation_metrics)

            log_images = batch_idx == 0 and self.current_epoch > 0 and self.is_ap_epoch()
            if log_images and isinstance(self.logger, pl.loggers.wandb.WandbLogger):
                channel_grids = self.visualize_predictions_channels(result_dict)
                self.log_channel_predictions_grids(channel_grids, mode="validation")

                keypoint_grids = self.visualize_predicted_keypoints(result_dict)
                self.log_predicted_keypoints(keypoint_grids, mode="validation")

        ## log (defaults to on_epoch, which aggregates the logged values over entire validation set)
        self.log("validation/epoch_loss", result_dict["loss"])
        self.log("validation/gt_loss", result_dict["gt_loss"])

    def test_step(self, test_batch, batch_idx):
        # no need to switch model to eval mode, this is handled by pytorch lightning
        result_dict = self.shared_step(test_batch, batch_idx, include_visualization_data_in_result_dict=True)
        self.update_ap_metrics(result_dict, self.ap_test_metrics)
        # only log first 10 batches to reduce storage space
        if batch_idx < 10 and isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            image_grids = self.visualize_predictions_channels(result_dict)
            self.log_channel_predictions_grids(image_grids, mode="test")

            keypoint_grids = self.visualize_predicted_keypoints(result_dict)
            self.log_predicted_keypoints(keypoint_grids, mode="test")

        self.log("test/epoch_loss", result_dict["loss"])
        self.log("test/gt_loss", result_dict["gt_loss"])

    def log_and_reset_mean_ap(self, mode: str):
        mean_ap_per_threshold = torch.zeros(len(self.maximal_gt_keypoint_pixel_distances))
        if mode == "train":
            metrics = self.ap_training_metrics
        elif mode == "validation":
            metrics = self.ap_validation_metrics
        elif mode == "test":
            metrics = self.ap_test_metrics
        else:
            raise ValueError(f"mode {mode} not recognized")

        # calculate APs for each channel and each threshold distance, and log them
        print(f" # {mode} metrics:")
        for channel_idx, channel_name in enumerate(self.keypoint_channel_configuration):
            channel_aps = self.compute_and_log_metrics_for_channel(metrics[channel_idx], channel_name, mode)
            mean_ap_per_threshold += torch.tensor(channel_aps)

        # calculate the mAP over all channels for each threshold distance, and log them
        for i, maximal_distance in enumerate(self.maximal_gt_keypoint_pixel_distances):
            self.log(
                f"{mode}/meanAP/d={float(maximal_distance):.1f}",
                mean_ap_per_threshold[i] / len(self.keypoint_channel_configuration),
            )

        # calculate the mAP over all channels and all threshold distances, and log it
        mean_ap = mean_ap_per_threshold.mean() / len(self.keypoint_channel_configuration)
        self.log(f"{mode}/meanAP", mean_ap)
        self.log(f"{mode}/meanAP/meanAP", mean_ap)

    def training_epoch_end(self, outputs):
        """
        Called on the end of a training epoch.
        Used to compute and log the AP metrics.
        """
        if self.is_ap_epoch():
            self.log_and_reset_mean_ap("train")

    def validation_epoch_end(self, outputs):
        """
        Called on the end of a validation epoch.
        Used to compute and log the AP metrics.
        """
        if self.is_ap_epoch():
            self.log_and_reset_mean_ap("validation")

    def test_epoch_end(self, outputs):
        """
        Called on the end of a test epoch.
        Used to compute and log the AP metrics.
        """
        self.log_and_reset_mean_ap("test")

    def update_channel_ap_metrics(
        self, predicted_heatmaps: torch.Tensor, gt_keypoints: List[torch.Tensor], validation_metric: KeypointAPMetrics
    ):
        """
        Updates the AP metric for a batch of heatmaps and keypoins of a single channel (!)
        This is done by extracting the detected keypoints for each heatmap and combining them with the gt keypoints for the same frame, so that
        the confusion matrix can be determined together with the distance thresholds.

        predicted_heatmaps: N x H x W tensor with the batch of predicted heatmaps for a single channel
        gt_keypoints: List of size N, containing K_i x 2 tensors with the ground truth keypoints for the channel of that sample
        """

        # log corner keypoints to AP metrics for all images in this batch
        formatted_gt_keypoints = [
            [Keypoint(int(k[0]), int(k[1])) for k in frame_gt_keypoints] for frame_gt_keypoints in gt_keypoints
        ]
        batch_detected_channel_keypoints = self.extract_detected_keypoints_from_heatmap(
            predicted_heatmaps.unsqueeze(1)
        )
        batch_detected_channel_keypoints = [batch_detected_channel_keypoints[i][0] for i in range(len(gt_keypoints))]
        for i, detected_channel_keypoints in enumerate(batch_detected_channel_keypoints):
            validation_metric.update(detected_channel_keypoints, formatted_gt_keypoints[i])

    def compute_and_log_metrics_for_channel(
        self, metrics: KeypointAPMetrics, channel: str, training_mode: str
    ) -> List[float]:
        """
        logs AP of predictions of single Channel for each threshold distance.
        Also resets metric and returns resulting AP for all distances.
        """
        ap_metrics = metrics.compute()
        rounded_ap_metrics = {k: round(v, 3) for k, v in ap_metrics.items()}
        print(f"{channel} : {rounded_ap_metrics}")
        for maximal_distance, ap in ap_metrics.items():
            self.log(f"{training_mode}/{channel}_ap/d={float(maximal_distance):.1f}", ap)

        mean_ap = sum(ap_metrics.values()) / len(ap_metrics.values())
        self.log(f"{training_mode}/{channel}_ap/meanAP", mean_ap)  # log top level for wandb hyperparam chart.

        metrics.reset()
        return list(ap_metrics.values())

    def is_ap_epoch(self) -> bool:
        """Returns True if the AP should be calculated in this epoch."""
        is_epch = self.ap_epoch_start <= self.current_epoch and self.current_epoch % self.ap_epoch_freq == 0
        # always log the AP in the last epoch
        is_epch = is_epch or self.current_epoch == self.trainer.max_epochs - 1

        # if user manually specified a validation frequency, we should always log the AP in that epoch
        is_epch = is_epch or (self.current_epoch > 0 and self.trainer.check_val_every_n_epoch > 1)
        return is_epch

    def extract_detected_keypoints_from_heatmap(self, heatmap: torch.Tensor) -> List[DetectedKeypoint]:
        """
        Extract keypoints from a single channel prediction and format them for AP calculation.

        Args:
        heatmap (torch.Tensor) : H x W tensor that represents a heatmap.
        """
        if heatmap.dtype == torch.float16:
            # Maxpool_2d not implemented for FP16 apparently
            heatmap_to_extract_from = heatmap.float()
        else:
            heatmap_to_extract_from = heatmap

        keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
            heatmap_to_extract_from, self.max_keypoints, self.minimal_keypoint_pixel_distance, return_scores=True
        )
        detected_keypoints = [
            [[] for _ in range(heatmap_to_extract_from.shape[1])] for _ in range(heatmap_to_extract_from.shape[0])
        ]
        for batch_idx in range(len(detected_keypoints)):
            for channel_idx in range(len(detected_keypoints[batch_idx])):
                for kp_idx in range(len(keypoints[batch_idx][channel_idx])):
                    detected_keypoints[batch_idx][channel_idx].append(
                        DetectedKeypoint(
                            keypoints[batch_idx][channel_idx][kp_idx][0],
                            keypoints[batch_idx][channel_idx][kp_idx][1],
                            scores[batch_idx][channel_idx][kp_idx],
                        )
                    )

        return detected_keypoints
