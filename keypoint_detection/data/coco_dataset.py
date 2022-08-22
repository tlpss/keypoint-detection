import argparse
from ast import parse
from collections import defaultdict
import json
from os import stat
from pathlib import Path
from sys import maxunicode
from typing import List, Union
import typing

import torch
from torchvision.transforms import ToTensor

from keypoint_detection.data.utils import ImageDataset, ImageLoader, IOSafeImageLoaderDecorator
from keypoint_detection.utils.tensor_padding import pad_tensor_with_nans
from keypoint_detection.data.json_formats.coco_parser import CocoImage, CocoKeypointCategory, CocoKeypoints

class COCOKeypointsDataset(ImageDataset):


    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        The default values here are actually duplicates from the init function, but this was for readability (??)
        """
        parser = parent_parser.add_argument_group("BlenderkeypointsDataset")
        parser.add_argument(
            "--json_dataset_path",
            required=False,
            type=str,
            help="Absolute path to the json file that defines the dataset according to the COCO format.",
        )
        return parent_parser
    def __init__(self, json_dataset_path: str, keypoint_channels: list[list], imageloader: ImageLoader = None):
        super().__init__(imageloader)


        self.to_tensor_transform = ToTensor()
        self.dataset_json_path = Path(json_dataset_path)
        self.dataset_path =  self.dataset_json_path.parent

        self.keypoint_channels =  keypoint_channels

        self.dataset = self.prepare_dataset() # idx: (image, list(keypoints/channel))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image_path =  self.dataset_path / self.dataset[index][0]
        image = self.image_loader.get_image(str(image_path), index)
        image = self.to_tensor_transform(image)

        keypoints_per_channel = self.dataset[index][1]

        return image, keypoints_per_channel

    def prepare_dataset(self):
        with open(self.dataset_json_path,"r") as file:
            data = json.load(file)
            parsed_coco = CocoKeypoints(**data)

            # iterate over all annotations and create a dict {img_id: {semantic_type : [keypoints]}}
            # make sure to deal with multiple occurances of same semantic_type in one image (e.g. multipe humans in one image)

            img_dict: typing.Dict[int, CocoImage] = {}
            for img in parsed_coco.images:
                img_dict[img.id] = img
            
            category_dict: typing.Dict[int, CocoKeypointCategory] = {}
            for category in parsed_coco.categories:
                category_dict[category.id] = category
            
            annotation_dict = defaultdict(lambda: defaultdict(lambda: [])) 
            for annotation in parsed_coco.annotations:
                img = img_dict[annotation.image_id]
                category = category_dict[category.id]
                semantic_classes = category.keypoints

                keypoints = annotation.keypoints
                keypoints = self.split_list_in_keypoints(keypoints)
                for semantic_type, keypoint in zip(semantic_classes, keypoints):
                    #rescale to pixel coordinates
                    keypoint[0] *= img.width
                    keypoint[1] *= img.height
                    annotation_dict[annotation.image_id][semantic_type].append(keypoint)

            # iterate over each image and all it's annotations
            # filter the visible keypoints
            # and group them by channel 
            dataset = []
            for img_id, keypoint_dict in annotation_dict.items():
                img_channels_keypoints = [[] for _ in range(len(self.keypoint_channels))]
                for semantic_type, keypoints in keypoint_dict.items():
                    for keypoint in keypoints:
                        if self.is_keypoint_visible(keypoint):
                            channel_idx = self.get_keypoint_channel_index(semantic_type)
                            if channel_idx > -1:
                                img_channels_keypoints[channel_idx].append(keypoint[:2])

                dataset.append([img.file_name, img_channels_keypoints])
            
            return dataset
            
    def get_keypoint_channel_index(self,semantic_type:str) -> int:

        for i,types_in_channel in enumerate(self.keypoint_channels):
            if semantic_type in types_in_channel:
                return i
        return -1 

    def is_keypoint_visible(self, keypoint):
        # TODO: make unvisible but annotated keypoints selectable.
        return keypoint[2] > 1
            
    @staticmethod
    def split_list_in_keypoints(list_to_split):
        n = 3
        output=[list_to_split[i:i + n] for i in range(0, len(list_to_split), n)]
        return output
        
    @staticmethod
    def collate_fn(data):
        """ custom collate function for torch dataloader
        that padds (NaNs) all lists of keypoints to the maximum number of keypoints in a single channel for the current batch.

        Note that it could have been more efficient to padd for each channel separately, but it's not worth the trouble as even 
        for 100 channels with each 100 occurances the padded data size is still < 1kB..

        Args:
            data: list of tuples (image, keypoints); image = 3xHxW tensor; keypoints = List(c x list(? keypoints ))

        Returns:
            (images, keypoints) Nx3xHxW, NxCxmax_keypoints_for_any_channel_in_batch x 2; where all are padded with NaNs.
        """
        images, keypoints = zip(*data)

        # get max amount of keypoints in any channel for the given batch
        max_num_keypoints = max(max(len(x) for x in y) for y in keypoints)
        #padd all keypoints with NaNs and create a single tensor out of them.
        keypoints = [[pad_tensor_with_nans(torch.tensor(x),max_num_keypoints) for x in y] for y in keypoints]
        keypoints = torch.stack([torch.stack(x) for x in keypoints])
        
        images = torch.stack(images)

        return images, keypoints


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = COCOKeypointsDataset(Path(__file__).parents[2] / "test"  /"test_dataset" / "coco_dataset.json",[["box_corner0","box_corner3","flap_corner0"],["box_corner12"]])
    dataloader = DataLoader(dataset,2, collate_fn= dataset.collate_fn)
    img, keypoints = dataset[1]
    print(dataset[1])
    print(len(dataset))
    print(dataloader)
    imgs, keypoints  =next(iter(dataloader))
    print(imgs.shape)
    print(keypoints.shape)
