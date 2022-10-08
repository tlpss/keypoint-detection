import shutil
import unittest
from pathlib import Path

import numpy as np

from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from labeling.scripts.crop_coco_dataset import create_cropped_dataset

from .configuration import DEFAULT_HPARAMS


class TestCropCocoDataset(unittest.TestCase):
    def test_crop_coco_dataset(self):
        annotations_filename = "coco_dataset.json"
        input_json_dataset_path = Path(__file__).parents[0] / "test_dataset" / annotations_filename
        output_dataset_path = create_cropped_dataset(input_json_dataset_path, 32, 32)
        print(output_dataset_path)

        output_json_dataset_path = Path(output_dataset_path) / annotations_filename

        # Check whether the new coords are half of the old, because image resolution was halved.
        channel_config = DEFAULT_HPARAMS["keypoint_channel_configuration"]
        dataset_old = COCOKeypointsDataset(input_json_dataset_path, channel_config)
        dataset_new = COCOKeypointsDataset(output_json_dataset_path, channel_config)

        for item_old, item_new in zip(dataset_old, dataset_new):
            _, keypoint_channels_old = item_old
            _, keypoint_channels_new = item_new

            for channel_old, channel_new in zip(keypoint_channels_old, keypoint_channels_new):
                for keypoint_old, keypoint_new in zip(channel_old, channel_new):
                    print(keypoint_old, keypoint_new)
                    assert np.allclose(np.array(keypoint_old) / 2.0, np.array(keypoint_new))

        shutil.rmtree(output_dataset_path)
