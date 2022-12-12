import os

import torch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_PARAMS = {
    "image_size": 64,
    "dataset_size": 4,
}

DEFAULT_HPARAMS = {
    "keypoint_channel_configuration": [["box_corner0", "box_corner1", "box_corner2", "box_corner3"], ["flap_corner0"]],
    "detect_non_visible_keypoints": True,
    "seed": 102,
    "wandb_project": "test_project",
    "wandb_entity": "box-manipulation",
    "max_epochs": 10,
    "log_every_n_steps": 2,
    "gpus": 1 if torch.cuda.is_available() else 0,
    "json_dataset_path": os.path.join(TEST_DIR, "test_dataset/coco_dataset.json"),
    "image_dataset_path": os.path.join(TEST_DIR, "test_dataset/images"),
    "batch_size": 2,
    "validation_split_ratio": 0.25,
    "num_workers": 2,
    "backbone_type": "DilatedCnn",
    "learning_rate": 3e-4,
    "lr_scheduler_relative_threshold": 0.01,
    "ap_epoch_start": 4,
    "ap_epoch_freq": 2,
    "heatmap_sigma": 1,
    "minimal_keypoint_extraction_pixel_distance": 1,
    "maximal_gt_keypoint_pixel_distances": "2 4",
    "early_stopping_relative_threshold": 0.01,
    "n_channels_in": 3,
    "n_downsampling_layers": 2,
    "n_resnet_blocks": 2,
    "n_channels": 4,
    "kernel_size": 3,
    "dilation": 1,
    "random-arg-to-test": "random",
    "max_keypoints": 20
}
