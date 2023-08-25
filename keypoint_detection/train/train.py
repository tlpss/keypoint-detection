from argparse import ArgumentParser
from typing import List, Tuple

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

from keypoint_detection.data.datamodule import KeypointsDataModule
from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.train.utils import create_pl_trainer
from keypoint_detection.utils.path import get_wandb_log_dir_path


def add_system_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """
    function that adds all system configuration (hyper)parameters to the provided argumentparser
    """
    parser = parent_parser.add_argument_group("System")
    parser.add_argument("--seed", default=2022, help="seed for reproducibility")
    parser.add_argument("--wandb_project", default="test-project", help="The wandb project to log the results to")
    parser.add_argument(
        "--wandb_entity",
        default="airo-box-manipulation",
        help="The entity name to log the project against, can be simply set to your username if you have no dedicated entity for this project",
    )
    parser.add_argument(
        "--keypoint_channel_configuration",
        type=str,
        help="A list of the semantic keypoints that you want to learn in each channel. These semantic categories must be defined in the COCO dataset. Seperate the channels with a ; and the categories within a channel with a =",
    )

    parser.add_argument(
        "--early_stopping_relative_threshold",
        default=0.0,
        type=float,
        help="relative threshold for early stopping callback. If validation epoch loss does not increase with at least this fraction compared to the best result so far for 5 consecutive epochs, training is stopped.",
    )
    # deterministic argument for PL trainer, not exposed in their CLI.
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
    # set to True by default, but can be set to False to speed up training.

    parser.add_argument(
        "--non-deterministic-pytorch",
        action="store_false",
        dest="deterministic",
        help="do not use deterministic algorithms for pytorch. This can speed up training, but will make it non-reproducible.",
    )

    parser.set_defaults(deterministic=True)
    return parent_parser


def main(hparams: dict) -> Tuple[KeypointDetector, pl.Trainer]:
    """
    Initializes the datamodule, model and trainer based on the global hyperparameters.
    calls trainer.fit(model, module) afterwards and returns both model and trainer.
    """
    # seed all random number generators on all processes and workers for reproducibility
    pl.seed_everything(hparams["seed"], workers=True)

    # use deterministic algorithms for torch to ensure exact reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    # this can slow down training
    # but the impact is limited in my experience.
    # so I prefer to be deterministic (and hence reproducible) by default.

    # also note that following is not enough:
    # torch.backends.cudnn.deterministic = True
    # there are other non-deterministic algorithms
    # cf list at https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    # the following is still not good enough with Pytorch-Lightning:
    # import torch
    # torch.use_deterministic_algorithms(True)
    # though I am not exactly sure why.
    # so we have to set it in the trainer! (see create_pl_trainer)

    backbone = BackboneFactory.create_backbone(**hparams)
    model = KeypointDetector(backbone=backbone, **hparams)
    data_module = KeypointsDataModule(**hparams)
    wandb_logger = WandbLogger(
        project=hparams["wandb_project"],
        entity=hparams["wandb_entity"],
        save_dir=get_wandb_log_dir_path(),
        log_model="all",  # log all checkpoints made by PL, see create_trainer for callback
    )
    trainer = create_pl_trainer(hparams, wandb_logger)
    trainer.fit(model, data_module)

    if "json_test_dataset_path" in hparams:
        # check if we have a best checkpoint, if not, use the current weights but log a warning
        # it makes more sense to evaluate on the best checkpoint because, i.e. the best validation score obtained.
        # evaluating on the current weights is more noisy and would also result in lower evaluation scores if overfitting happens
        #  when training longer, even with perfect i.i.d. test/val sets. This is not desired.

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            print("No best checkpoint found, using current weights for test set evaluation")
        trainer.test(model, data_module, ckpt_path="best")

    return model, trainer


def parse_channel_configuration(channel_configuration: str) -> List[List[str]]:
    assert isinstance(channel_configuration, str)
    channels = channel_configuration.split(";")
    channels = [[category.strip() for category in channel.split("=")] for channel in channels]
    return channels


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
    parser = KeypointsDataModule.add_argparse_args(parser)
    parser = BackboneFactory.add_to_argparse(parser)

    # get parser arguments and filter the specified arguments
    hparams = vars(parser.parse_args())
    # remove the unused optional items without default, which have None as key
    hparams = {k: v for k, v in hparams.items() if v is not None}
    hparams["keypoint_channel_configuration"] = parse_channel_configuration(hparams["keypoint_channel_configuration"])
    print(f" argparse arguments ={hparams}")

    # initialize wandb here, this allows for using wandb sweeps.
    # with sweeps, wandb will send hyperparameters to the current agent after the init
    # these can then be found in the 'config'
    # (so wandb params > argparse)
    wandb.init(
        project=hparams["wandb_project"],
        entity=hparams["wandb_entity"],
        config=hparams,
        dir=get_wandb_log_dir_path(),  # dir should already exist! will fallback to /tmp and not log images otherwise..
    )

    # get (possibly updated by sweep) config parameters
    hparams = wandb.config
    print(f" config after wandb init: {hparams}")

    print("starting training")
    main(hparams)
