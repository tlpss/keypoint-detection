import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn

from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.metrics import DetectedKeypoint, Keypoint, KeypointAPMetrics
from keypoint_detection.utils.heatmap import generate_keypoints_heatmap, get_keypoints_from_heatmap
from keypoint_detection.utils.tensor_padding import unpad_nans_from_tensor
from keypoint_detection.utils.visualization import visualize_predictions


class KeypointDetector(pl.LightningModule):
    """
    keypoint Detector using Gaussian Heatmaps
    There can be N channels of keypoints, each with its own set of ground truth keypoints.
    The mean Average precision is used to calculate the performance.

    """

    @staticmethod
    def add_model_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        """
        parser = parent_parser.add_argument_group("KeypointDetector")

        # TODO: add these with inspection to avoid manual duplication!

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
            default=10,
            help="Epoch at which to start calculating the AP every `ap_epoch_frequency` epochs.",
        )
        parser.add_argument(
            "--ap_epoch_freq",
            type=int,
            default=10,
            help="Rate at which to calculate the AP metric if epoch > `ap_epoch_start`",
        )

        return parent_parser

    def __init__(
        self,
        heatmap_sigma: int,
        maximal_gt_keypoint_pixel_distances: Union[str, List[float]],
        minimal_keypoint_extraction_pixel_distance: int,
        learning_rate: float,
        backbone: Backbone,
        loss_function,
        keypoint_channels: Union[str, List[str]],
        ap_epoch_start,
        ap_epoch_freq,
        **kwargs,
    ):
        """[summary]

        Args:
            heatmap_sigma (int, optional): Sigma of the gaussian heatmaps used to train the detector. Defaults to 10.
            n_channels (int, optional): Number of channels for the CNN layers. Defaults to 32.
            detect_flap_keypoints (bool, optional): Detect flap keypoints in a second channel or use a single channel Detector for box corners only.
            minimal_keypoint_extraction_pixel_distance (int, optional): the minimal distance (in pixels) between two detected keypoints,
                                                                        or the size of the local mask in which a keypoint needs to be the local maximum
            maximal_gt_keypoint_pixel_distance (int, optional): the maximal distance between a gt keypoint and detected keypoint, for the keypoint to be considered a TP
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
        self.heatmap_loss = loss_function

        if isinstance(keypoint_channels, list):
            self.keypoint_channels = keypoint_channels
        else:
            self.keypoint_channels = keypoint_channels.strip().split(" ")

        if isinstance(maximal_gt_keypoint_pixel_distances, str):
            maximal_gt_keypoint_pixel_distances = [
                float(val) for val in maximal_gt_keypoint_pixel_distances.strip().split(" ")
            ]

            self.maximal_gt_keypoint_pixel_distances = maximal_gt_keypoint_pixel_distances
        else:
            self.maximal_gt_keypoint_pixel_distances = maximal_gt_keypoint_pixel_distances

        self.ap_validaiton_metric = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channels
        ]

        self.n_channels_out = len(self.keypoint_channels)

        head = nn.Conv2d(
            in_channels=backbone.get_n_channels_out(),
            out_channels=self.n_channels_out,
            kernel_size=(3, 3),
            padding="same",
        )

        self.model = nn.Sequential(
            backbone,
            head,
            nn.Sigmoid(),  # create probabilities
        )

        # save hyperparameters to logger, to make sure the model hparams are saved even if
        # they are not included in the config (i.e. if they are kept at the defaults).
        # this is for later reference and consistency.
        self.save_hyperparameters(ignore="**kwargs")

    def forward(self, x: torch.Tensor):
        """
        x shape must be (N,C_in,H,W) with N batch size, and C_in number of incoming channels (3)
        return shape = (N, C_out, H,W)
        """
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        return optimizer

    def shared_step(self, batch, batch_idx, validate=False) -> Dict[str, Any]:
        """
        shared step for train and validation step

        batch: img, keypoints
        where img is a B,3,H,W tensor
        and keypoints a list of len(channels) with B,N,(2/3) keypoints for each channel

        returns:

        shared_dict (Dict): a dict with the heatmaps, gt_keypoints and losses
        """
        imgs, padded_keypoints = batch
        channel_keypoints = []
        channel_gt_heatmaps = []
        for channel_idx in range(len(self.keypoint_channels)):
            num_kp = padded_keypoints[channel_idx].shape[0]
            unpadded_kp = [unpad_nans_from_tensor(padded_keypoints[channel_idx][j, :, :]) for j in range(num_kp)]
            channel_keypoints.append(unpadded_kp)
            channel_gt_heatmaps.append(self.create_heatmap_batch(imgs[0].shape[1:], channel_keypoints[channel_idx]))
        imgs = imgs.to(self.device)

        ## predict and compute losses
        predicted_heatmaps = self.forward(imgs)

        channel_losses = []
        channel_gt_losses = []

        result_dict = {}
        for channel_idx in range(len(self.keypoint_channels)):
            channel_losses.append(
                self.heatmap_loss(predicted_heatmaps[:, channel_idx, :, :], channel_gt_heatmaps[channel_idx])
            )
            channel_gt_losses.append(
                self.heatmap_loss(channel_gt_heatmaps[channel_idx], channel_gt_heatmaps[channel_idx])
            )  # BCE gt loss does not go to zero.. (entropy)

            # pass losses and other info to result dict
            result_dict.update({f"{self.keypoint_channels[channel_idx]}_loss": channel_losses[channel_idx].detach()})
            if validate:
                result_dict.update(
                    {f"{self.keypoint_channels[channel_idx]}_keypoints": channel_keypoints[channel_idx]}
                )

        # only pass predictions in validate step to avoid overhead in train step.
        if validate:
            result_dict.update({"predicted_heatmaps": predicted_heatmaps.detach()})

        loss = sum(channel_losses)
        gt_loss = sum(channel_gt_losses)
        result_dict.update({"loss": loss, "gt_loss": gt_loss})

        # visualization
        if batch_idx == 0 and self.current_epoch > 0:
            for channel_idx, channel_name in enumerate(self.keypoint_channels):
                visualize_predictions(
                    imgs,
                    predicted_heatmaps[:, channel_idx, :, :].detach(),
                    channel_gt_heatmaps[channel_idx],
                    self.logger,
                    self.minimal_keypoint_pixel_distance,
                    channel_name,
                    validate=validate,
                )

        return result_dict

    def training_step(self, train_batch, batch_idx):

        result_dict = self.shared_step(train_batch, batch_idx)

        for channel_name in self.keypoint_channels:
            self.log(f"train/{channel_name}", result_dict[f"{channel_name}_loss"])

        self.log("train/loss", result_dict["loss"])
        self.log("train/gt_loss", result_dict["gt_loss"])
        return result_dict

    def validation_step(self, val_batch, batch_idx):

        result_dict = self.shared_step(val_batch, batch_idx, validate=True)

        if self.is_ap_epoch():
            # update corner AP metric
            for channel_idx, channel_name in enumerate(self.keypoint_channels):
                predicted_channel_heatmaps = result_dict["predicted_heatmaps"][:, channel_idx, :, :]
                gt_corner_keypoints = result_dict[f"{channel_name}_keypoints"]
                self.update_ap_metrics(
                    predicted_channel_heatmaps, gt_corner_keypoints, self.ap_validaiton_metric[channel_idx]
                )

        ## log (defaults to on_epoch, which aggregates the logged values over entire validation set)
        self.log("validation/epoch_loss", result_dict["loss"])
        self.log("validation/gt_loss", result_dict["gt_loss"])

    def validation_epoch_end(self, outputs):
        """
        Called on the end of the validation epoch.
        Used to compute and log the AP metrics.
        """

        if self.is_ap_epoch():
            mean_ap = 0.0
            for channel_idx, channel_name in enumerate(self.keypoint_channels):
                mean_ap += self.compute_and_log_metrics(self.ap_validaiton_metric[channel_idx], channel_name)

            mean_ap /= len(self.keypoint_channels)

            self.log("meanAP", mean_ap)

    ##################
    # util functions #
    ##################
    @classmethod
    def get_artifact_dir_path(cls) -> Path:
        return Path(__file__).resolve().parents[1] / "artifacts"

    @classmethod
    def get_wand_log_dir_path(cls) -> Path:
        return Path(__file__).resolve().parents[1] / "wandb"

    def update_ap_metrics(
        self, predicted_heatmaps: torch.Tensor, gt_keypoints: torch.Tensor, validation_metric: KeypointAPMetrics
    ):
        """
        Update provided AP metric by extracting the detected keypoints for each heatmap
        and combining them with the gt keypoints for the same frame
        """
        # log corner keypoints to AP metrics, frame by frame
        formatted_gt_keypoints = [
            [Keypoint(int(k[0]), int(k[1])) for k in frame_gt_keypoints] for frame_gt_keypoints in gt_keypoints
        ]
        for i, predicted_frame_heatmap in enumerate(torch.unbind(predicted_heatmaps, 0)):
            detected_corner_keypoints = self.extract_detected_keypoints(predicted_frame_heatmap)
            validation_metric.update(detected_corner_keypoints, formatted_gt_keypoints[i])

    def compute_and_log_metrics(self, validation_metric: KeypointAPMetrics, channel: str) -> float:
        """
        logs ap for each max_distance, resets metric and returns meanAP
        """
        # compute ap's
        ap_metrics = validation_metric.compute()
        print(f"{ap_metrics=}")
        for maximal_distance, ap in ap_metrics.items():
            self.log(f"validation/{channel}_ap/d={maximal_distance}", ap)

        mean_ap = sum(ap_metrics.values()) / len(ap_metrics.values())

        self.log(f"validation/{channel}_meanAP", mean_ap)  # log top level for wandb hyperparam chart.
        validation_metric.reset()
        return mean_ap

    def is_ap_epoch(self):
        return (
            self.ap_epoch_start <= self.current_epoch
            and self.current_epoch % self.ap_epoch_freq == 0
            or self.current_epoch == self.trainer.max_epochs - 1
        )

    def create_heatmap_batch(self, shape: Tuple[int, int], keypoints: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            shape (Tuple): H,W
            keypoints (torch.Tensor): N x K x 3 Tensor with batch of keypoints.

        Returns:
            (torch.Tensor): N x H x W Tensor with N heatmaps
        """

        batch_heatmaps = [
            generate_keypoints_heatmap(shape, keypoints[i], self.heatmap_sigma, self.device)
            for i in range(len(keypoints))
        ]
        batch_heatmaps = torch.stack(batch_heatmaps, dim=0)
        return batch_heatmaps

    def extract_detected_keypoints(self, heatmap: torch.Tensor) -> List[DetectedKeypoint]:
        """
        get keypoints of single channel from single frame.

        Args:
        heatmap (torch.Tensor) : B x H x W tensor that represents a heatmap.
        """

        detected_keypoints = get_keypoints_from_heatmap(heatmap, self.minimal_keypoint_pixel_distance)
        keypoint_probabilities = self.compute_keypoint_probability(heatmap, detected_keypoints)
        detected_keypoints = [
            DetectedKeypoint(detected_keypoints[i][0], detected_keypoints[i][1], keypoint_probabilities[i])
            for i in range(len(detected_keypoints))
        ]

        return detected_keypoints

    def compute_keypoint_probability(
        self, heatmap: torch.Tensor, detected_keypoints: List[Tuple[int, int]]
    ) -> List[float]:
        """Compute probability measure for each detected keypoint on the heatmap

        Args:
            heatmap: Heatmap
            detected_keypoints: List of extreacted keypoints

        Returns:
            : [description]
        """
        return [heatmap[k[0]][k[1]].item() for k in detected_keypoints]
