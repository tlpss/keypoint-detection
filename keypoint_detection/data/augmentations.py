from typing import List
import typing
from unittest import result
import albumentations as A
import numpy as np 
from keypoint_detection.types import KEYPOINT_TYPE, CHANNEL_KEYPOINTS_TYPE, IMG_KEYPOINTS_TYPE
from typing import Tuple

class MultiChannelKeypointsCompose(A.Compose):
    """A subclass of Albumentations.Compose to accomodate for multiple groups/channels of keypoints. 
    Albumentations offers to discard keypoints that become invisible after a transform.. 
    The problem was that albumentations expects a single list of keypoints. You can add a label to each keypoint, but the labels are already 
    discarded after the keypoint_detection.data.COCOKeypointsDataset init function as the keypoints are then grouped per channel.. and so it was easier to just add a subclass to handle
 for allchannels in the way they are structured in the dataset: [[u,v],[u,v]],[[u,v]]]

    This is done by simply flattening the keypoints, then passing them all to Albumentations and then unflattening them back while discarding those 
    that are no longer visible.
    """
    def __init__(self, transforms, bbox_params = None, keypoint_params = None, additional_targets= None, p: float = 1, final_image_size = 256):
        super().__init__(transforms, bbox_params, keypoint_params, additional_targets, p)
        self.final_image_size = 256

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
        """Wraps Albumentations transforms to deal with multiple-channel keypoints. There is no way to know how many keypoints are left for each 
        channel afterwards so we select the visible keypoints ourselves within each channel separately.

        Args:
            force_apply (bool, optional): _description_. Defaults to False.

        Returns:
            typing.Dict[str, typing.Any]: _description_
        """
        keypoints = data["keypoints"]
        flattened_keypoints = self.flatten_keypoints(keypoints)
        data["keypoints"] = flattened_keypoints
        result_dict = super().__call__(*args, force_apply=force_apply, **data)
        transformed_flattened_keypoints = result_dict["keypoints"]
        transformed_keypoints = self.order_transformed_keypoints_per_channel_and_remove_invisible(keypoints, transformed_flattened_keypoints)
        result_dict["keypoints"] = transformed_keypoints
        return result_dict


    @staticmethod
    def flatten_keypoints(keypoints: IMG_KEYPOINTS_TYPE) -> List[KEYPOINT_TYPE]:
        return [item for sublist in keypoints for item in sublist]
    
    @staticmethod
    def is_visible(u,v,width,height):
        visible = 0 <= u and u <  width 
        visible = visible and (0 <= v and v < height)
        return visible
    
    def order_transformed_keypoints_per_channel_and_remove_invisible(self,original_keypoints: IMG_KEYPOINTS_TYPE, transformed_keypoints: List[KEYPOINT_TYPE]) -> IMG_KEYPOINTS_TYPE:
        ordered_transformed_keypoints = [[] for list in original_keypoints]
        idx = 0
        for channel_idx,channel_keypoints in enumerate(original_keypoints):
            for _ in channel_keypoints:
                kp = transformed_keypoints[idx]
                if self.is_visible(kp[0],kp[1], self.final_image_size, self.final_image_size):
                    ordered_transformed_keypoints[channel_idx].append(kp)
                idx += 1
        return ordered_transformed_keypoints