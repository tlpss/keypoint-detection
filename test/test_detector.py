import unittest

import torch
from pytorch_lightning.loggers import WandbLogger

from keypoint_detection.data.datamodule import RandomSplitDataModule
from keypoint_detection.data.dataset import KeypointsDataset
from keypoint_detection.models.backbones.unet import UnetBackbone
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.loss import bce_loss
from keypoint_detection.models.metrics import KeypointAPMetric
from keypoint_detection.train.utils import create_pl_trainer
from keypoint_detection.utils.heatmap import generate_keypoints_heatmap

from .configuration import DEFAULT_HPARAMS


class TestHeatmapUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.image_width = 32
        self.image_height = 16
        self.keypoints = [[10, 4], [10, 8], [30, 7]]
        self.sigma = 3

        self.heatmaps = generate_keypoints_heatmap(
            (self.image_height, self.image_width), self.keypoints, self.sigma, "cpu"
        )
        self.loss_function = bce_loss
        self.hparams = DEFAULT_HPARAMS
        self.backbone = UnetBackbone(**self.hparams)
        self.model = KeypointDetector(backbone=self.backbone, loss_function=self.loss_function, **self.hparams)

        self.module = RandomSplitDataModule(KeypointsDataset(**self.hparams), **self.hparams)

    def test_perfect_heatmap(self):
        loss = self.model.heatmap_loss(self.heatmaps, self.heatmaps)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss >= 0)

    def test_heatmap_batch(self):
        batch_tensor = torch.Tensor([self.keypoints, self.keypoints])
        print(batch_tensor.shape)
        batch_heatmap = self.model.create_heatmap_batch((self.image_height, self.image_width), batch_tensor)
        self.assertEqual(batch_heatmap.shape, (2, self.image_height, self.image_width))


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.loss_function = bce_loss
        self.hparams = DEFAULT_HPARAMS
        self.backbone = UnetBackbone(**self.hparams)
        self.model = KeypointDetector(backbone=self.backbone, loss_function=self.loss_function, **self.hparams)

        self.module = RandomSplitDataModule(KeypointsDataset(**self.hparams), **self.hparams)

    def test_shared_step_batch(self):

        model = self.model

        batch = next(iter(self.module.train_dataloader()))

        result_dict = model.shared_step(batch, 0)

        assert result_dict["loss"]
        assert result_dict["gt_loss"]
        assert result_dict[f"corner_keypoints_loss"]

    def test_train(self):
        """
        run train and evaluation to see if all goes as expected
        """
        wandb_logger = WandbLogger(dir=KeypointDetector.get_wandb_log_dir_path(), mode="offline")

        model = self.model

        trainer = create_pl_trainer(self.hparams, wandb_logger)
        trainer.fit(model, self.module)

        batch = next(iter(self.module.train_dataloader()))
        imgs, keypoints = batch
        with torch.no_grad():
            model(imgs)

    def test_gt_heatmaps(self):
        max_dst = 2
        metric = KeypointAPMetric(max_dst)

        for batch in self.module.train_dataloader():
            imgs, keypoints = batch
            heatmaps = self.model.create_heatmap_batch(imgs[0].shape[1:], keypoints[0])
            self.model.update_ap_metrics(heatmaps, keypoints[0], metric)

        ap = metric.compute()
        self.assertEqual(ap, 1.0)

    def test_initial_values(self):
        detector = self.model

        random_batch = torch.randn(1, 3, 100, 100)

        heatmap = detector(random_batch)

        self.assertTrue(torch.mean(heatmap).item() < 0.1)
        self.assertTrue(torch.var(heatmap).item() < 0.1)


# TODO: test model train script
