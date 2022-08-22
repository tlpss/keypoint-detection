import json
import unittest
from pathlib import Path

from keypoint_detection.data.json_formats.coco import CocoKeypoints


class TestCocoParser(unittest.TestCase):
    # def test_valid_coco_json(self):
    #     path = Path(__file__).parent / "person_keypoints_val2017.json"
    #     with open(path, "r") as file:
    #         data = json.load(file)
    #         CocoKeypoints(**data)

    def test_example_coco_json(self):
        path = Path(__file__).parents[1] / "test_dataset" / "dataset.json"
        with open(path, "r") as file:
            data = json.load(file)
            CocoKeypoints(**data)
