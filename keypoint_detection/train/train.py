from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

from keypoint_detection.data.datamodule import RandomSplitDataModule
from keypoint_detection.data.dataset import KeypointsDatasetPreloaded
from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.loss import LossFactory
from keypoint_detection.train.utils import create_pl_trainer_from_args


def add_system_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """
    function that adds all system configuration (hyper)parameters to the provided argumentparser
    """
    parser = parent_parser.add_argument_group("System")
    parser.add_argument("--seed", default=2022)
    parser.add_argument("--wandb_project", default="test-project")
    parser.add_argument("--wandb_entity", default="airo-box-manipulation")
    parser.add_argument(
        "--keypoint_channels",
        type=str,
        help="The names of the keypoint channels that you want to detect, as they are defined in the dataset.json file",
    )
    parser.add_argument(
        "--keypoint_channel_max_keypoints",
        type=str,
        help="The maximal number of keypoints within each channel, used to pad the keypoint tensor if the number of (visible) keypoints is not constant",
    )
    return parent_parser


def main(hparams: dict) -> Tuple[KeypointDetector, pl.Trainer]:
    """
    Initializes the datamodule, model and trainer based on the global hyperparameters.
    calls trainer.fit(model, module) afterwards and returns both model and trainer.
    """
    pl.seed_everything(hparams["seed"], workers=True)

    backbone = BackboneFactory.create_backbone(**hparams)
    loss = LossFactory.create_loss(**hparams)
    model = KeypointDetector(backbone=backbone, loss_function=loss, **hparams)

    dataset = KeypointsDatasetPreloaded(**hparams)

    module = RandomSplitDataModule(dataset, **hparams)
    wandb_logger = WandbLogger(
        project=hparams["wandb_project"],
        entity=hparams["wandb_entity"],
        dir=KeypointDetector.get_wand_log_dir_path(),
        log_model=True,
    )
    trainer = create_pl_trainer_from_args(hparams, wandb_logger)
    trainer.fit(model, module)
    return model, trainer


if __name__ == "__main__":
    """
    1. creates argumentparser with Model, Trainer and system paramaters; which can be used to overwrite default parameters
    when running python train.py --<param> <param_value>
    2. sets ups wandb and loads the local config in wandb.
    3. pulls the config from the wandb instance, which allows wandb to update this config when a sweep is used to set some config parameters
    4. calls the main function to start the training process
    """

    # create the parser, add module arguments and the system arguments
    parser = ArgumentParser()
    parser = add_system_args(parser)
    parser = KeypointDetector.add_model_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = KeypointsDatasetPreloaded.add_argparse_args(parser)
    parser = RandomSplitDataModule.add_argparse_args(parser)
    parser = BackboneFactory.add_to_argparse(parser)
    parser = LossFactory.add_to_argparse(parser)

    # get parser arguments and filter the specified arguments
    hparams = vars(parser.parse_args())
    # remove the unused optional items without default, which have None as key
    hparams = {k: v for k, v in hparams.items() if v is not None}
    print(f" argparse arguments ={hparams}")

    # initialize wandb here, this allows for using wandb sweeps.
    # with sweeps, wandb will send hyperparameters to the current agent after the init
    # these can then be found in the 'config'
    # (so wandb params > argparse)
    wandb.init(
        project=hparams["wandb_project"],
        entity=hparams["wandb_entity"],
        config=hparams,
        dir=KeypointDetector.get_wand_log_dir_path(),
    )

    # get (possibly updated by sweep) config parameters
    hparams = wandb.config
    print(f" config after wandb init: {hparams}")

    print("starting trainig")
    main(hparams)
