import inspect

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from keypoint_detection.models.detector import KeypointDetector


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

    early_stopping = EarlyStopping(monitor="validation/epoch_loss", patience=5, verbose=True, mode="min")
    trainer = pl.Trainer(**trainer_kwargs, callbacks=[early_stopping])
    return trainer
