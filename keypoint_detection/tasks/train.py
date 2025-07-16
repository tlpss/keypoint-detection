"""train detector based on argparse configuration"""
from argparse import ArgumentParser
from typing import Tuple
import os

import pytorch_lightning as pl
import wandb
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

from keypoint_detection.data.datamodule import KeypointsDataModule
from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.tasks.train_utils import create_pl_trainer, parse_channel_configuration
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from keypoint_detection.utils.path import get_wandb_log_dir_path, get_artifact_dir_path

def add_trainer_args(parser):
    """Add trainer-specific arguments to the parser."""
    trainer_group = parser.add_argument_group("Trainer")

    # Core training arguments
    trainer_group.add_argument("--max_epochs", type=int, default=100)
    trainer_group.add_argument("--min_epochs", type=int, default=1)

    # Hardware/performance
    trainer_group.add_argument("--accelerator", type=str, default="auto")
    trainer_group.add_argument("--devices", type=str, default="auto")
    trainer_group.add_argument("--precision", type=str, default="32-true")

    # Validation settings
    trainer_group.add_argument("--check_val_every_n_epoch", type=int, default=1)
    trainer_group.add_argument("--val_check_interval", type=float, default=1.0)

    # Checkpointing
    trainer_group.add_argument("--enable_checkpointing", action="store_true", default=True)

    # Logging
    trainer_group.add_argument("--log_every_n_steps", type=int, default=50)
    trainer_group.add_argument("--enable_progress_bar", action="store_true", default=True)

    return parser
def add_system_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """
    function that adds all system configuration (hyper)parameters to the provided argumentparser
    """
    parser = parent_parser.add_argument_group("System")
    parser.add_argument("--seed", default=2022, help="seed for reproducibility")
    parser.add_argument(
        "--wandb_project", default="keypoint-detection", help="The wandb project to log the results to"
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="The entity name to log the project against, can be simply set to your username if you have no dedicated entity for this project",
    )
    parser.add_argument(
        "--wandb_name", default=None, help="The name of the run, if not specified, a random name will be generated"
    )
    parser.add_argument(
        "--keypoint_channel_configuration",
        type=str,
        help="A list of the semantic keypoints that you want to learn in each channel. These semantic categories must be defined in the COCO dataset. Seperate the channels with a : and the categories within a channel with a =",
    )

    parser.add_argument(
        "--early_stopping_relative_threshold",
        default=-1.0,  # no early stopping by default
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

    parser.add_argument(
        "--wandb_checkpoint_artifact",
        type=str,
        help="A checkpoint to resume/start training from. keep in mind that you currently cannot specify hyperparameters other than the LR.",
        required=False,
    )
    parser.set_defaults(deterministic=True)
    return parent_parser


def train(hparams: dict) -> Tuple[KeypointDetector, pl.Trainer]:
    """
    Initializes the datamodule, model and trainer based on the global hyperparameters.
    calls trainer.fit(model, module) afterwards and returns both model and trainer.
    """
    # seed all random number generators on all processes and workers for reproducibility
    pl.seed_everything(hparams["seed"], workers=True)

    # use deterministic algorithms for torch to ensure exact reproducibility
    # we have to set it in the trainer! (see create_pl_trainer)
    if hparams["wandb_checkpoint_artifact"] is not None:
        print("Loading checkpoint from wandb")
        # This will create a KeypointDetector model with the associated hyperparameters.
        # Model weights will be loaded.
        # Optimizer and LR scheduler will be initiated from scratch" (if you want to really resume training, you have to pass the ckeckpoint to the trainer)
        # cf. https://lightning.ai/docs/pytorch/latest/common/checkpointing_basic.html#lightningmodule-from-checkpoint
        model = get_model_from_wandb_checkpoint(hparams["wandb_checkpoint_artifact"])
        # TODO: how can specific hparams be overwritten here? e.g. LR reduction for finetuning or something?
    else:
        backbone = BackboneFactory.create_backbone(**hparams)
        model = KeypointDetector(backbone=backbone, **hparams)

    data_module = KeypointsDataModule(**hparams)
    wandb_logger = WandbLogger(
        # these are already set in the wandb init (to start from a sweep config)
        # name=hparams["wandb_name"],
        # project=hparams["wandb_project"],
        # entity=hparams["wandb_entity"],
        save_dir=get_wandb_log_dir_path(),
        log_model=True,  # only log checkpoints at the end of training, i.e. only log the best checkpoint
        # not suitable for expensive training runs where you might want to restart from checkpoint
        # but this saves storage and usually keypoint detector training runs are not that expensive anyway
    )
    hparams['strategy'] = "ddp_find_unused_parameters_false" if int(hparams["devices"]) > 1 else "auto"
    trainer = create_pl_trainer(hparams, wandb_logger)
    #model = torch.compile(model)
    trainer.fit(model, data_module)

    if "json_test_dataset_path" in hparams and hparams.get("json_test_dataset_path"):
        # check if we have a best checkpoint, if not, use the current weights but log a warning
        # it makes more sense to evaluate on the best checkpoint because, i.e. the best validation score obtained.
        # evaluating on the current weights is more noisy and would also result in lower evaluation scores if overfitting happens
        #  when training longer, even with perfect i.i.d. test/val sets. This is not desired.

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            print("No best checkpoint found, using current weights for test set evaluation")
            ckpt_path = None
        else:
            print(f"Using best checkpoint for test set evaluation: {ckpt_path}")
            ckpt_path = "best"
        trainer.test(model, data_module, ckpt_path=ckpt_path)

    return model, trainer

def train_cli():
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
    parser = add_trainer_args(parser)
    parser = KeypointsDataModule.add_argparse_args(parser)
    parser = BackboneFactory.add_to_argparse(parser)

    # get parser arguments and filter the specified arguments
    hparams = vars(parser.parse_args())
    hparams["keypoint_channel_configuration"] = parse_channel_configuration(hparams["keypoint_channel_configuration"])
    
    os.environ["WANDB_DIR"] = get_wandb_log_dir_path()
    os.environ["PL_ARTIFACT_DIR"] = get_artifact_dir_path()

    # initialize wandb here, this allows for using wandb sweeps.
    # with sweeps, wandb will send hyperparameters to the current agent after the init
    # these can then be found in the 'config'
    # (so wandb params > argparse)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.init(
            name=hparams["wandb_name"],
            project=hparams["wandb_project"],
            entity=hparams["wandb_entity"],
            config=hparams,
            dir=os.environ["WANDB_DIR"],
        )
    else:
        # For other ranks, set to offline mode
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

    # get (possibly updated by sweep) config parameters
    #hparams = wandb.config
    hparams = {**hparams, **wandb.config}

    train(hparams)


if __name__ == "__main__":
    train_cli()
