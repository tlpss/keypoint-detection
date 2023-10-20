"""run evaluation on a model for the given dataset"""


import argparse

import pytorch_lightning as pl
import torch

from keypoint_detection.data.datamodule import KeypointsDataModule
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint


def evaluate_model(model: KeypointDetector, datamodule: KeypointsDataModule) -> None:
    """evaluate the model on the given datamodule and checkpoint path"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        deterministic=True,
    )
    output = trainer.test(model, datamodule)
    return output


def eval_cli():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--wandb_checkpoint", type=str, required=True, help="The wandb checkpoint to load the model from"
    )
    argparser.add_argument(
        "--test_json_path",
        type=str,
        required=True,
        help="The path to the json file that defines the test dataset according to the COCO format.",
    )
    args = argparser.parse_args()

    wandb_checkpoint = args.wandb_checkpoint
    test_json_path = args.test_json_path

    model = get_model_from_wandb_checkpoint(wandb_checkpoint)
    data_module = KeypointsDataModule(
        model.keypoint_channel_configuration, json_test_dataset_path=test_json_path, batch_size=8
    )
    evaluate_model(model, data_module)


if __name__ == "__main__":

    wandb_checkpoint = "tlips/synthetic-cloth-keypoints-single-towel/model-gl39yjtf:v0"
    test_json_path = "/home/tlips/Documents/synthetic-cloth-data/synthetic-cloth-data/data/datasets/TOWEL/07-purple-towel-on-white/annotations_val.json"
    test_json_path = "/storage/users/tlips/aRTFClothes/cloth-on-white/purple-towel-on-white_resized_512x256/purple-towel-on-white.json"
    model = get_model_from_wandb_checkpoint(wandb_checkpoint)
    data_module = KeypointsDataModule(
        model.keypoint_channel_configuration, json_test_dataset_path=test_json_path, batch_size=8
    )
    output = evaluate_model(model, data_module)
