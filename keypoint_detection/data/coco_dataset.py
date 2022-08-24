import argparse
import json
import typing
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from torchvision.transforms import ToTensor

from keypoint_detection.data.coco_parser import CocoImage, CocoKeypointCategory, CocoKeypoints
from keypoint_detection.data.imageloader import ImageDataset, ImageLoader


class COCOKeypointsDataset(ImageDataset):
    """Pytorch Dataset for COCO-formatted Keypoint dataset

    cf. https://cocodataset.org/#format-data for more information. We expect each annotation to have at least the keypoints and num_keypoints fields.
    Each category should also have keypoints. For more information on the required fields and data types, have a look at the COCO parser in `coco_parser.py`.

    The dataset builds an index during the init call that maps from each image_id to a list of all keypoints of all semantic types in the dataset.

    The Dataset also expects a keypoint_channel_configuration that maps from the semantic types (the keypoints in all categories of the COCO file) to the channels
    of the keypoint detector. In the simplest case this is simply a list of all types, but for e.g. symmetric objects or equivalence mapping one could combine different
    types into one channel. For example if you have category box with keypoints [corner0, corner1, corner2, corner3] you could combine  them in a single channel for the
    detector by passing as configuration [[corner0,corner1,corner2,corner3]].

    You can also select if you want to train on annotations with flag=1 (occluded).

    The paths in the JSON should be relative to the directory in which the JSON is located.


    The __getitem__ function returns [img_path, [keypoints for each channel according to the configuration]]
    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        """
        parser = parent_parser.add_argument_group("COCOkeypointsDataset")
        parser.add_argument(
            "--json_dataset_path",
            required=True,
            type=str,
            help="Absolute path to the json file that defines the dataset according to the COCO format.",
        )
        parser.add_argument(
            "--detect_non_visible_keypoints",
            default=True,
            type=str,
            help="detect keypoints with visibility flag = 1? default = True",
        )

        return parent_parser

    def __init__(
        self,
        json_dataset_path: str,
        keypoint_channel_configuration: list[list[str]],
        detect_non_visible_keypoints: bool = True,
        imageloader: ImageLoader = None,
        **kwargs
    ):
        super().__init__(imageloader)

        self.image_to_tensor_transform = ToTensor()
        self.dataset_json_path = Path(json_dataset_path)
        self.dataset_dir_path = self.dataset_json_path.parent  # assume paths in JSON are relative to this directory!

        self.keypoint_channel_configuration = keypoint_channel_configuration
        self.detect_non_visible_keypoints = detect_non_visible_keypoints

        self.dataset = self.prepare_dataset()  # idx: (image, list(keypoints/channel))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns:
            (image, keypoints); image = 3xHxW tensor; keypoints = List(c x list( list of K_i keypoints ))

            e.g. for 2 heatmap channels with respectively 1,2 keypoints, the keypoints list will be formatted as
            [[[u11,v11,f11]],[[u21,v21,f21],[u22,v22,f22]]]
        """
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image_path = self.dataset_dir_path / self.dataset[index][0]
        image = self.image_loader.get_image(str(image_path), index)
        image = self.image_to_tensor_transform(image)

        keypoints_per_channel = self.dataset[index][1]

        return image, keypoints_per_channel

    def prepare_dataset(self):
        """Prepares the dataset to map from COCO to (img, [keypoints for each channel])

        Returns:
            [img_path, [list of keypoints for each channel]]
        """
        with open(self.dataset_json_path, "r") as file:
            data = json.load(file)
            parsed_coco = CocoKeypoints(**data)

            img_dict: typing.Dict[int, CocoImage] = {}
            for img in parsed_coco.images:
                img_dict[img.id] = img

            category_dict: typing.Dict[int, CocoKeypointCategory] = {}
            for category in parsed_coco.categories:
                category_dict[category.id] = category

            # iterate over all annotations and create a dict {img_id: {semantic_type : [keypoints]}}
            # make sure to deal with multiple occurances of same semantic_type in one image (e.g. multipe humans in one image)
            annotation_dict = defaultdict(
                lambda: defaultdict(lambda: [])
            )  # {img_id: {channel : [keypoints for that channel]}}
            for annotation in parsed_coco.annotations:
                # add all keypoints from this annotation to the corresponding image in the dict

                img = img_dict[annotation.image_id]
                category = category_dict[category.id]
                semantic_classes = category.keypoints

                keypoints = annotation.keypoints
                keypoints = self.split_list_in_keypoints(keypoints)
                for semantic_type, keypoint in zip(semantic_classes, keypoints):
                    # rescale to pixel coordinates
                    keypoint[0] *= img.width
                    keypoint[1] *= img.height
                    annotation_dict[annotation.image_id][semantic_type].append(keypoint)

            # iterate over each image and all it's annotations
            # filter the visible keypoints
            # and group them by channel
            dataset = []
            for img_id, keypoint_dict in annotation_dict.items():
                img_channels_keypoints = [[] for _ in range(len(self.keypoint_channel_configuration))]
                for semantic_type, keypoints in keypoint_dict.items():
                    for keypoint in keypoints:
                        if self.is_keypoint_visible(keypoint):
                            channel_idx = self.get_keypoint_channel_index(semantic_type)
                            if channel_idx > -1:
                                img_channels_keypoints[channel_idx].append(keypoint[:2])

                dataset.append([img_dict[img_id].file_name, img_channels_keypoints])

            return dataset

    def get_keypoint_channel_index(self, semantic_type: str) -> int:
        """
        given a semantic type, get it's channel according to the channel configuration.
        Returns -1 if the semantic type couldn't be found.
        """

        for i, types_in_channel in enumerate(self.keypoint_channel_configuration):
            if semantic_type in types_in_channel:
                return i
        return -1

    def is_keypoint_visible(self, keypoint: List) -> bool:
        """
        Args:
            keypoint (list): [u,v,flag]

        Returns:
            bool: True if current keypoint is considered visible according to the dataset configuration, else False
        """
        minimal_flag = 0
        if not self.detect_non_visible_keypoints:
            minimal_flag = 1
        return keypoint[2] > minimal_flag

    @staticmethod
    def split_list_in_keypoints(list_to_split: List) -> List[List]:
        """
        splits list [u1,v1,f1,u2,v2,f2,...] to [[u,v,f],..]
        """
        n = 3
        output = [list_to_split[i : i + n] for i in range(0, len(list_to_split), n)]
        return output

    @staticmethod
    def collate_fn(data):
        """custom collate function for use with the torch dataloader

        Note that it could have been more efficient to padd for each channel separately, but it's not worth the trouble as even
        for 100 channels with each 100 occurances the padded data size is still < 1kB..

        Args:
            data: list of tuples (image, keypoints); image = 3xHxW tensor; keypoints = List(c x list(? keypoints ))

        Returns:
            (images, keypoints); Images as a torch tensor Nx3xHxW,
            keypoints is a nested list of lists. where each item is a tensor (K,2) with K the number of keypoints
            for that channel and that sample:

                List(List(Tensor(K,2))) -> C x N x Tensor(max_keypoints_for_any_channel_in_batch x 2)

        Note there is no padding, as all values need to be unpacked again in the detector to create all the heatmaps,
        unlike e.g. NLP where you directly feed the padded sequences to the network.
        """
        images, keypoints = zip(*data)

        # convert the list of keypoints to a 2D tensor
        keypoints = [[torch.tensor(x) for x in y] for y in keypoints]
        # reorder to have the different keypoint channels as  first dimension
        # C x N x K x 2 , K = variable number of keypoints for each (N,C)
        reordered_keypoints = [[keypoints[i][j] for i in range(len(keypoints))] for j in range(len(keypoints[0]))]

        images = torch.stack(images)

        return images, reordered_keypoints
