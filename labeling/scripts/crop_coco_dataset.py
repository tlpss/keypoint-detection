import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import albumentations as A
import cv2

from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.data.coco_parser import CocoKeypoints
from keypoint_detection.data.imageloader import ImageLoader, IOSafeImageLoaderDecorator


def save_cropped_image_and_edit_annotations(
    i, image_info, image_annotations, height_new, width_new, image_loader, input_dataset_path, output_dataset_path
):
    input_image_path = os.path.join(input_dataset_path, image_info.file_name)
    image = image_loader.get_image(input_image_path, i)

    min_size = min(image.shape[0], image.shape[1])
    transform = A.Compose(
        [
            A.CenterCrop(min_size, min_size),
            A.Resize(height_new, width_new),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    # Extract keypoints to the format albumentations wants.
    image_keypoints = []
    for annotation in image_annotations:
        annotation_keypoints = COCOKeypointsDataset.split_list_in_keypoints(annotation.keypoints)
        for keypoint in annotation_keypoints:
            image_keypoints.append(keypoint[:2])
    keypoints_xy = [keypoint[:2] for keypoint in image_keypoints]

    # Transform image and keypoints
    transformed = transform(image=image, keypoints=keypoints_xy)
    transformed_image = transformed["image"]
    transformed_keypoints = transformed["keypoints"]

    # Edit the original keypoints.
    index = 0
    for annotation in image_annotations:
        for i in range(len(annotation.keypoints) // 3):
            annotation.keypoints[3 * i : 3 * i + 2] = transformed_keypoints[index]
            index += 1

    # Save transformed image to disk
    output_image_path = os.path.join(output_dataset_path, image_info.file_name)
    image_directory = os.path.dirname(output_image_path)
    os.makedirs(image_directory, exist_ok=True)
    image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, image_bgr)


def create_cropped_dataset(input_json_dataset_path, height_new, width_new):
    input_dataset_path = os.path.dirname(input_json_dataset_path)
    output_dataset_path = input_dataset_path + f"_{height_new}x{width_new}"

    if os.path.exists(output_dataset_path):
        print(f"{output_dataset_path} exists, quiting.")
        return

    with open(input_json_dataset_path, "r") as file:
        data = json.load(file)
        parsed_coco = CocoKeypoints(**data)

    image_loader = IOSafeImageLoaderDecorator(ImageLoader())
    annotations = parsed_coco.annotations

    images_annotations = defaultdict(list)
    for annotation in annotations:
        print(type(annotation))
        images_annotations[annotation.image_id].append(annotation)

    for i, image_info in enumerate(parsed_coco.images):
        image_annotations = images_annotations[image_info.id]
        save_cropped_image_and_edit_annotations(
            i,
            image_info,
            image_annotations,
            height_new,
            width_new,
            image_loader,
            input_dataset_path,
            output_dataset_path,
        )

    annotations_json = os.path.join(output_dataset_path, os.path.basename(input_json_dataset_path))
    with open(annotations_json, "w") as file:
        json.dump(parsed_coco.dict(exclude_none=True), file)

    return output_dataset_path


if __name__ == "__main__":
    """
    example usage:

    python crop_coco_dataset.py datasets/towel_testset_0 256 256

    This will create a new dataset called towel_testset_0_256x256 in the same directory as the old one.
    The old dataset will be unaltered.
    Currently only square outputs are supported.
    """

    parser = ArgumentParser()
    parser.add_argument("input_json_dataset_path")
    parser.add_argument("height_new", type=int)
    parser.add_argument("width_new", type=int)
    args = parser.parse_args()
    create_cropped_dataset(**vars(args))
