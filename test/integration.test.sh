#!/bin/bash
# This is an integration test for the keypoint detector
# Run from the repo's root folder using bash test/integration_test.sh

python keypoint_detection/train/train.py --keypoint_channels  "corner_keypoints flap_corner_keypoints" \
--keypoint_channel_max_keypoints "-1 -1" --image_dataset_path "test/test_dataset" \
--json_dataset_path "test/test_dataset/coc_dataset.json" --batch_size  1 --wandb_project "keypoint-detector-integration-test" \
--max_epochs 50 --early_stopping_relative_threshold -1.0 --log_every_n_steps 1
