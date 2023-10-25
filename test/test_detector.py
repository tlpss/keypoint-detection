import os
import unittest

import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from keypoint_detection.data.datamodule import KeypointsDataModule
from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from keypoint_detection.models.backbones.unet import Unet
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.metrics import KeypointAPMetric
from keypoint_detection.tasks.train_utils import create_pl_trainer
from keypoint_detection.utils.heatmap import create_heatmap_batch, generate_channel_heatmap
from keypoint_detection.utils.load_checkpoints import load_from_checkpoint
from keypoint_detection.utils.path import get_wandb_log_dir_path

from .configuration import DEFAULT_HPARAMS


class TestHeatmapUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.image_width = 32
        self.image_height = 16
        self.keypoints = torch.Tensor([[10, 4], [10, 8], [30, 7]])
        self.sigma = 3

        self.heatmaps = generate_channel_heatmap(
            (self.image_height, self.image_width), self.keypoints, self.sigma, "cpu"
        )
        self.hparams = DEFAULT_HPARAMS
        self.backbone = Unet(**self.hparams)
        self.model = KeypointDetector(backbone=self.backbone, **self.hparams)

        self.module = KeypointsDataModule(**self.hparams)

    def test_perfect_heatmap_loss(self):
        loss = nn.functional.binary_cross_entropy(self.heatmaps, self.heatmaps)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss >= 0)
        # loss is not zero! (bce)


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.hparams = DEFAULT_HPARAMS
        self.backbone = BackboneFactory.create_backbone(**self.hparams)
        self.model = KeypointDetector(backbone=self.backbone, **self.hparams)

        self.module = KeypointsDataModule(**self.hparams)

    def test_shared_step_batch(self):

        model = self.model
        batch = next(iter(self.module.train_dataloader()))
        result_dict = model.shared_step(batch, 0)
        assert result_dict["loss"]
        assert isinstance(result_dict["loss"], torch.Tensor)
        assert result_dict["gt_loss"]

    def test_train(self):
        """
        run train and evaluation to see if all goes as expected
        """
        wandb_logger = WandbLogger(dir=get_wandb_log_dir_path(), mode="offline")

        model = self.model

        trainer = create_pl_trainer(self.hparams, wandb_logger)
        trainer.fit(model, self.module)

        batch = next(iter(self.module.train_dataloader()))
        imgs, keypoints = batch
        with torch.no_grad():
            model(imgs)

    def test_ap_calculation_on_gt_heatmaps(self):
        max_dst = 2
        metric = KeypointAPMetric(max_dst)

        for batch in self.module.train_dataloader():
            imgs, keypoints = batch
            heatmaps = create_heatmap_batch(
                imgs[0].shape[1:], keypoints[0], self.model.heatmap_sigma, self.model.device
            )
            self.model.update_channel_ap_metrics(heatmaps, keypoints[0], metric)

        ap = metric.compute()
        self.assertEqual(ap, 1.0)

    def test_model_init_heatmaps(self):
        # should be low, to avoid hockey stick loss curve
        # since most of the heatmaps has to be low-probability
        detector = self.model

        random_batch = torch.randn(1, 3, 100, 100)

        heatmap = detector(random_batch)

        self.assertTrue(torch.mean(heatmap).item() < 0.1)
        self.assertTrue(torch.var(heatmap).item() < 0.1)

    def test_checkpoint_loading(self):
        wandb_logger = WandbLogger(dir=get_wandb_log_dir_path(), mode="offline")

        model = self.model
        trainer = create_pl_trainer(self.hparams, wandb_logger)
        trainer.fit(model, self.module)

        trainer.save_checkpoint("test.ckpt")
        checkpointed_model = load_from_checkpoint("test.ckpt")
        os.remove("test.ckpt")
        assert checkpointed_model
