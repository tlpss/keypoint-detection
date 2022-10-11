import typing
from typing import List

import albumentations as A

from keypoint_detection.types import IMG_KEYPOINTS_TYPE, KEYPOINT_TYPE


class MultiChannelKeypointsCompose(A.Compose):
    """A subclass of Albumentations.Compose to accomodate for multiple groups/channels of keypoints.
    Some transforms (crop e.g.) will result in certain keypoints no longer being in the new image. Albumentations can remove them, but since it operates
    on a single list of keypoints, the transformed keypoints need to be associated to their channel afterwards. Albumentations has support for labels to accomodate this,
    so we label each keypoint with the index of its channel.
    """

    def __init__(self, transforms, p: float = 1):
        keypoint_params = A.KeypointParams(format="xy", label_fields=["channel_labels"], remove_invisible=True)
        super().__init__(transforms, keypoint_params=keypoint_params, p=p)

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:

        # flatten and create channel labels (=str(index))
        keypoints = data["keypoints"]
        self.create_channel_labels(keypoints)
        flattened_keypoints = self.flatten_keypoints(keypoints)
        data["keypoints"] = flattened_keypoints
        data["channel_labels"] = self.flatten_keypoints(self.create_channel_labels(keypoints))
        # apply transforms
        result_dict = super().__call__(*args, force_apply=force_apply, **data)

        # rearrange keypoints by channel
        transformed_flattened_keypoints = result_dict["keypoints"]
        transformed_flattened_labels = result_dict["channel_labels"]
        transformed_keypoints = self.order_transformed_keypoints_by_channel(
            keypoints, transformed_flattened_keypoints, transformed_flattened_labels
        )
        result_dict["keypoints"] = transformed_keypoints
        return result_dict

    @staticmethod
    def flatten_keypoints(keypoints: IMG_KEYPOINTS_TYPE) -> List[KEYPOINT_TYPE]:
        return [item for sublist in keypoints for item in sublist]

    @staticmethod
    def create_channel_labels(keypoints: IMG_KEYPOINTS_TYPE):
        channel_labels = [[str(i)] * len(keypoints[i]) for i in range(len(keypoints))]
        return channel_labels

    @staticmethod
    def order_transformed_keypoints_by_channel(
        original_keypoints: IMG_KEYPOINTS_TYPE,
        transformed_keypoints: List[KEYPOINT_TYPE],
        transformed_channel_labels: List[str],
    ) -> IMG_KEYPOINTS_TYPE:
        ordered_transformed_keypoints = [[] for _ in original_keypoints]
        for transformed_keypoint, channel_label in zip(transformed_keypoints, transformed_channel_labels):
            channel_idx = int(channel_label)
            ordered_transformed_keypoints[channel_idx].append(transformed_keypoint)

        return ordered_transformed_keypoints
