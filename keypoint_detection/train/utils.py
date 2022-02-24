import inspect
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from keypoint_detection.models.detector import KeypointDetector


class RelativeEarlyStopping(EarlyStopping):
    def __init__(
        self,
        monitor: Optional[str] = None,
        min_relative_delta: float = 0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
    ):
        super().__init__(monitor, -0.2, patience, verbose, mode, strict)
        self.min_relative_delta = min_relative_delta

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = ""

        # the * current is the only change.
        if self.monitor_op(current - self.min_delta * current, self.best_score.to(current.device)):
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


def create_pl_trainer_from_args(hparams: dict, wandb_logger: WandbLogger) -> Trainer:
    """
    function that creates a pl.Trainer instance from the given global hyperparameters and logger
    and adds the specified callbacks.
    """

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: hparams[name] for name in valid_kwargs if name in hparams}
    trainer_kwargs.update({"logger": wandb_logger})
    # set dir for checkpoints
    trainer_kwargs.update({"default_root_dir": KeypointDetector.get_artifact_dir_path()})

    early_stopping = RelativeEarlyStopping(
        monitor="validation/epoch_loss", patience=5, min_relative_delta=0.001, verbose=True, mode="min"
    )
    trainer = pl.Trainer(**trainer_kwargs, callbacks=[early_stopping])
    return trainer
