import inspect
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from keypoint_detection.utils.path import get_artifact_dir_path


class RelativeEarlyStopping(EarlyStopping):
    """Slightly modified Early Stopping Callback that allows to set a relative threshold (loss * threshold)
    hope this will be integrated in Lightning one day: https://github.com/Lightning-AI/lightning/issues/12094


    """

    def __init__(
        self,
        monitor: Optional[str] = None,
        min_relative_delta: float = 0.01,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
    ):
        super().__init__(monitor, min_relative_delta, patience, verbose, mode, strict)

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = ""

        # the (1-delta)* current is the only change.
        if self.monitor_op(current * (1 - self.min_delta), self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: torch.Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved {abs(self.best_score - current)/self.best_score:.3f} times >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.7f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.7f}"
        return msg


def create_pl_trainer(hparams: dict, wandb_logger: WandbLogger) -> Trainer:
    """
    function that creates a pl.Trainer instance from the given global hyperparameters and logger
    and adds some callbacks.
    """

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: hparams[name] for name in valid_kwargs if name in hparams}
    trainer_kwargs.update({"logger": wandb_logger})
    # set dir for checkpoints
    trainer_kwargs.update({"default_root_dir": get_artifact_dir_path()})

    early_stopping = RelativeEarlyStopping(
        monitor="validation/epoch_loss",
        patience=5,
        min_relative_delta=float(hparams["early_stopping_relative_threshold"]),
        verbose=True,
        mode="min",
    )
    # cf https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.wandb.html

    # would be better to use mAP metric for checkpointing, but this is not calculated every epoch because it is rather expensive
    # epoch_loss still correlates rather well though
    # only store the best checkpoint and only the weights
    # so cannot be used to resume training but only for inference
    # saves storage though and training the detector is usually cheap enough to retrain it from scratch if you need specific weights etc.
    checkpoint_callback = ModelCheckpoint(
        monitor="validation/epoch_loss", mode="min", save_weights_only=True, save_top_k=1
    )

    trainer = pl.Trainer(**trainer_kwargs, callbacks=[early_stopping, checkpoint_callback])
    return trainer


def parse_channel_configuration(channel_configuration: str) -> List[List[str]]:
    assert isinstance(channel_configuration, str)
    channels = channel_configuration.split(":")
    channels = [[category.strip() for category in channel.split("=")] for channel in channels]
    return channels
