import inspect
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from keypoint_detection.models.detector import KeypointDetector


def create_pl_trainer_from_args(hparams: dict, wandb_logger: WandbLogger) -> Trainer:
    """
    function that creates a pl.Trainer instance from the given global hyperparameters and logger.

    pl only supports constructing from an Argparser or its output by default, but also allows to pass additional **kwargs.
    However, these kwargs must be present in the __init__ function, since ther is no additional **kwargs argument in the function
    to catch other kwargs (unlike in the Detector Module for example).
    Hence the global config dict is filtered to only include parameters that are present in the init function.

    To comply with the pl.Trainer.from_argparse_args declaration, an empty Nampespace (the result of argparser) is created and added to the call.
    """

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: hparams[name] for name in valid_kwargs if name in hparams}
    trainer_kwargs.update({"logger": wandb_logger})
    # set dir for checkpoints
    trainer_kwargs.update({"default_root_dir": KeypointDetector.get_artifact_dir_path()})

    # this call will add all relevant hyperparameters to the trainer, overwriting the empty namespace
    # and the default values in the Trainer class
    trainer = pl.Trainer.from_argparse_args(Namespace(), **trainer_kwargs)
    return trainer
