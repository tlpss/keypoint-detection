""" avoid circular imports by separating types"""
from typing import Tuple, List

KEYPOINT_TYPE = Tuple[int,int] #(u,v)
COCO_KEYPOINT_TYPE  =Tuple[int,int,int] # (u,v,f)
CHANNEL_KEYPOINTS_TYPE = List[KEYPOINT_TYPE]
IMG_KEYPOINTS_TYPE = List[CHANNEL_KEYPOINTS_TYPE]
