#!/bin/bash
# This is an integration test for the keypoint detector
# Run from the repo's root folder using bash test/integration_test.sh

python keypoint_detection/train/train.py \
--keypoint_channel_configuration  "box_corner0= box_corner1 = box_corner2= box_corner3; flap_corner0 ; flap_corner2" \
--json_dataset_path "test/test_dataset/coco_dataset.json" --batch_size  2 --wandb_project "keypoint-detector-integration-test" \
--max_epochs 50 --early_stopping_relative_threshold -1.0 --log_every_n_steps 1 --gpus 1
