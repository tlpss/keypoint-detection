import inspect
from argparse import ArgumentParser, Namespace
from typing import Tuple

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

from keypoint_detection.data.datamodule import RandomSplitDataModule
from keypoint_detection.data.dataset import KeypointsDatasetPreloaded
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.loss import LossFactory
from keypoint_detection.models.backbones.backbone_factory import BackboneFactory

default_config = {
    ## system params
    # Data params
    "image_dataset_path": f"{BoxDatasetPreloaded.get_data_dir_path()}/box_dataset2",
    "json_dataset_path": f"{BoxDatasetPreloaded.get_data_dir_path()}/box_dataset2/dataset.json",
    "batch_size": 4,
    "train_val_split_ratio": 0.1,
    # logging info
    "wandb_entity": "airo-box-manipulation",
    "wandb_project": "test-project",
    # Trainer params
    "seed": 2021,
    "max_epochs": 2,
    "gpus": 0,
    # model params -> default values in the model.
}


def add_system_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """
    function that adds all system configuration (hyper)parameters to the provided argumentparser
    """
    parser = parent_parser.add_argument_group("Trainer")
    parser.add_argument("--batch_size", required=False, type=int)
    parser.add_argument("--train_val_split_ratio", required=False, type=float)

    return parent_parser




def main(hparams: dict) -> Tuple[KeypointDetector, pl.Trainer]:
    """
    Initializes the datamodule, model and trainer based on the global hyperparameters.
    calls trainer.fit(model, module) afterwards and returns both model and trainer.
    """
    pl.seed_everything(hparams["seed"], workers=True)
    model = KeypointDetector(**hparams)

    dataset = BoxDatasetPreloaded(**hparams)

    module = BoxKeypointsDataModule(
        dataset,
        hparams["batch_size"],
        hparams["train_val_split_ratio"],
    )
    wandb_logger = WandbLogger(
        project=default_config["wandb_project"],
        entity=default_config["wandb_entity"],
        dir=KeypointDetector.get_wand_log_dir_path(),
        log_model=True,
    )
    trainer = create_pl_trainer_from_args(hparams, wandb_logger)
    trainer.fit(model, module)
    return model, trainer


if __name__ == "__main__":
    """
    1. loads default configuration parameters
    2. creates argumentparser with Model, Trainer and system paramaters; which can be used to overwrite default parameters
    when running python train.py --<param> <param_value>
    3. sets ups wandb and loads the local config in wandb.
    4. pulls the config from the wandb instance, which allows wandb to update this config when a sweep is used to set some config parameters
    5. calls the main function to start the training process
    """

    # start with the default config hyperparameters
    config = default_config

    # create the parser, add module arguments and the system arguments
    parser = ArgumentParser()
    parser = add_system_args(parser)
    parser = KeypointDetector.add_model_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = KeypointsDatasetPreloaded.add_argparse_args(parser)
    parser = BackboneFactory.add_to_argparse(parser)
    parser = LossFactory.add_to_argparse(parser)

    # get parser arguments and filter the specified arguments
    args = vars(parser.parse_args())


    # remove the unused optional items without default, which have None as key
    args = {k: v for k, v in args.items() if v is not None}  

    print(f" argparse arguments ={args}")

    # update the hyperparameters with the argparse parameters
    # this adds new <key,value> pairs if the keys did not exist and
    # updates the key with the new value pairs otherwise.
    # (so argparse > default)
    config.update(args)

    print(f" updated config parameters before wandb  = {config}")

    # initialize wandb here, this allows for using wandb sweeps.
    # with sweeps, wandb will send hyperparameters to the current agent after the init
    # these can then be found in the 'config'
    # (so wandb params > argparse > default)
    wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        config=config,
        dir=KeypointDetector.get_wand_log_dir_path(),
    )

    # get (possibly updated by sweep) config parameters
    config = wandb.config
    print(f" config after wandb init: {config}")

    # actual training.
    print("starting trainig")
    main(config)
