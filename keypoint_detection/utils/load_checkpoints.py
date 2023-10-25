from pathlib import Path

import torch

from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from keypoint_detection.models.detector import KeypointDetector


def get_model_from_wandb_checkpoint(checkpoint_reference: str):
    """
    get a model from a pytorch lightning checkpoint stored on wandb as artifact.

    checkpoint_reference: str e.g. 'airo-box-manipulation/iros2022_0/model-17tyvqfk:v3'
    """
    import wandb

    # download checkpoint locally (if not already cached)
    if wandb.run is None:
        run = wandb.init(project="inference")
    else:
        run = wandb.run
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    checkpoint_path = Path(artifact_dir) / "model.ckpt"
    return load_from_checkpoint(checkpoint_path)


def load_from_checkpoint(checkpoint_path: str, hparams_to_override: dict = None):
    """
    function to load a Keypoint Detector model from a local pytorch lightning checkpoint.

    These checkpoints contain everything that is need to continue training or to run inference.
    cf. https://pytorch-lightning.readthedocs.io/en/latest/common/checkpointing_basic.html?highlight=checkpoint#what-is-a-checkpoint

    checkpoint_path: path to the local checkpoint made by pytorch lightning, e.g. 'model.ckpt'
    """

    # load the hyperparameters from the checkpoint
    # to create the backbone using the backbone factory
    # all other arguments to the model are directly extracted from the checkpoint by pytorch lightning
    # but this is not possible for the backbone model, so we have to do it manually.
    # https://pytorch-lightning.readthedocs.io/en/latest/common/checkpointing_basic.html#save-hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    # create backbone and load checkpoint
    backbone = BackboneFactory.create_backbone(**checkpoint["hyper_parameters"])
    model = KeypointDetector.load_from_checkpoint(checkpoint_path, backbone=backbone)
    return model


if __name__ == "__main__":
    model = get_model_from_wandb_checkpoint("tlips/synthetic-cloth-keypoints-tshirts/model-4um302zo:v0")
    print(model.hparams)
