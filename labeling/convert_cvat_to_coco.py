from __future__ import annotations

import json
from typing import List

import tqdm

from keypoint_detection.data.coco_parser import CocoImage, CocoKeypointAnnotation, CocoKeypointCategory, CocoKeypoints
from labeling.file_loading import get_dict_from_json, get_dict_from_xml
from labeling.parsers.coco_categories_parser import COCOCategoriesConfig, COCOCategoryConfig
from labeling.parsers.cvat_keypoints_parser import CVATKeypointsParser, ImageItem, Point


def cvat_image_to_coco(
    cvat_xml_path: str, coco_category_configuration_path: str, image_folder: str = "images"
) -> dict:
    """Function that converts an annotation XML in the CVAT 1.1 Image format to the COCO keypoints format.

    This function supports:
    - multiple categories (box, tshirt);
    - multiple semantic types for each category ("corners", "flap_corners")
    - multiple keypoints for a single semantic type (a box has 4 corners) to facilitate fast labeling (no need to label each corner with a separate label, which requires geometric consistency)
    - occluded or invisible keypoints for each type

    It requires the CVAT dataset to be created by using labels formatted as <category>.<semantic_type>, using the group_id to group multiple instances together.
    if only a single instance is present, the group id is set to 1 by default so you don't have to do this yourself.

    To map from the CVAT labels to the COCO categories, you need to specify a configuration.
    See the readme for more details and an example.

    This function is rather complex unfortunately, but at a high level it performs the following:
    # for all categories in the config:
        # for all images:
            # create COCO Image
            # find number of  category instances in that images
            # for each instance in the image:
                # for all semantic types in the category:
                    # find all keypoints of that type for that instance in the current image
                # create a COCO Annotation for the current instance of the category

    Args:
        cvat_xml_path (str): _description_
        coco_category_configuration_path (str): _description_

    Returns:
        (dict): a COCO dict that can be dumped to a JSON.
    """
    cvat_dict = get_dict_from_xml(cvat_xml_path)
    cvat_parsed = CVATKeypointsParser(**cvat_dict)

    category_dict = get_dict_from_json(coco_category_configuration_path)
    parsed_category_config = COCOCategoriesConfig(**category_dict)

    # create a COCO Dataset Model
    coco_model = CocoKeypoints(images=[], annotations=[], categories=[])

    annotation_id_counter = 1  # counter for the annotation ID

    print("starting CVAT Image -> COCO conversion")
    for category in parsed_category_config.categories:
        print(f"converting category {category.name}")
        category_name = category.name
        category_keypoint_names = get_coco_keypoint_names_from_category_config(category)
        coco_model.categories.append(
            CocoKeypointCategory(
                id=category.id,
                name=category.name,
                supercategory=category.supercategory,
                keypoints=category_keypoint_names,
            )
        )

        for cvat_image in tqdm.tqdm(cvat_parsed.annotations.image):
            coco_image = CocoImage(
                file_name=f"{image_folder}/{cvat_image.name}",
                height=int(cvat_image.height),
                width=int(cvat_image.width),
                id=int(cvat_image.id) + 1,
            )
            coco_model.images.append(coco_image)
            n_image_category_instances = get_n_category_instances_in_image(cvat_image, category_name)
            for instance_id in range(1, n_image_category_instances + 1):  # IDs start with 1
                instance_category_keypoints = []
                for semantic_type in category.semantic_types:
                    keypoints = get_semantic_type_keypoints_from_instance_in_cvat_image(
                        cvat_image, semantic_type.name, instance_id
                    )

                    # pad for invisible keypoints for the given instance of the semantic type.
                    keypoints.extend([0.0] * (3 * semantic_type.n_keypoints - len(keypoints)))
                    instance_category_keypoints.extend(keypoints)

                coco_model.annotations.append(
                    CocoKeypointAnnotation(
                        category_id=category.id,
                        id=annotation_id_counter,
                        image_id=coco_image.id,
                        keypoints=instance_category_keypoints,
                    )
                )
                annotation_id_counter += 1
    return coco_model.dict(exclude_none=True)


### helper functions


def get_n_category_instances_in_image(cvat_image: ImageItem, category_name: str) -> int:
    """returns the number of instances for the specified category in the CVAT ImageItem.

    This is done by finding the maximum group_id for all annotations of the image.

    Edge cases include: no Points in the image or only 1 Point in the image.
    """
    if cvat_image.points is None:
        return 0
    if not isinstance(cvat_image.points, list):
        if get_category_from_cvat_label(cvat_image.points.label) == category_name:
            return int(cvat_image.points.group_id)
        else:
            return 0
    max_group_id = 1
    for cvat_point in cvat_image.points:
        if get_category_from_cvat_label(cvat_point.label) == category_name:
            max_group_id = max(max_group_id, int(cvat_point.group_id))
    return max_group_id


def get_category_from_cvat_label(label: str) -> str:
    """cvat labels are formatted as <category>.<semantic_type>
    this function returns the category
    """
    split = label.split(".")
    assert len(split) == 2, " label was not formatted as category.semantic_type"
    return label.split(".")[0]


def get_semantic_type_from_cvat_label(label: str) -> str:
    """cvat labels are formatted as <category>.<semantic_type>
    this function returns the semantic type
    """
    split = label.split(".")
    assert len(split) == 2, " label was not formatted as category.semantic_type"
    return label.split(".")[1]


def get_coco_keypoint_names_from_category_config(config: COCOCategoryConfig) -> List[str]:
    """Helper function that converts a CategoryConfiguration to a list of coco keypoints.
    This function duplicates keypoints for types with n_keypoints > 1 by appending an index:
    e.g. "corner", n_keypoints = 2 -> ["corner1" ,"corner2"].

    Args:
        config (dict): _description_

    Returns:
        _type_: _description_
    """
    keypoint_names = []
    for semantic_type in config.semantic_types:
        if semantic_type.n_keypoints == 1:
            keypoint_names.append(semantic_type.name)
        else:
            for i in range(semantic_type.n_keypoints):
                keypoint_names.append(f"{semantic_type.name}{i+1}")
    return keypoint_names


def get_semantic_type_keypoints_from_instance_in_cvat_image(
    cvat_image: ImageItem, semantic_type: str, instance_id: int
) -> List[float]:
    """Gather all keypoints of the given semantic type for this in the image.

    Args:
        cvat_image (ImageItem): _description_
        semantic_type (str): _description_
        instance_id (int): _description_

    Returns:
        List: _description_
    """
    instance_id = str(instance_id)
    if cvat_image.points is None:
        return [0.0, 0.0, 0]
    if not isinstance(cvat_image.points, list):
        if (
            semantic_type == get_semantic_type_from_cvat_label(cvat_image.points.label)
            and instance_id == cvat_image.points.group_id
        ):
            return extract_coco_keypoint_from_cvat_point(cvat_image.points, cvat_image)
        else:
            return [0.0, 0.0, 0]
    keypoints = []
    for cvat_point in cvat_image.points:
        if semantic_type == get_semantic_type_from_cvat_label(cvat_point.label) and instance_id == cvat_point.group_id:
            keypoints.extend(extract_coco_keypoint_from_cvat_point(cvat_point, cvat_image))
    return keypoints


def extract_coco_keypoint_from_cvat_point(cvat_point: Point, cvat_image: ImageItem) -> List:
    """extract keypoint in coco format (u,v,f) from cvat annotation point.
    Args:
        cvat_point (Point): _description_
        cvat_image (ImageItem): _description_

    Returns:
        List: [u,v,f] where u,v are the coords scaled to the image resolution and f is the coco visibility flag.
        see the coco dataset format for more details.
    """
    u = float(cvat_point.points.split(",")[0])
    v = float(cvat_point.points.split(",")[1])
    f = (
        1 if cvat_point.occluded == "1" else 2
    )  # occluded = 1 means not visible, which is 1 in COCO; visible in COCO is 2
    return [u, v, f]


if __name__ == "__main__":
    """
    example usage:

    python convert_cvat_to_coco.py --cvat_xml_file example/annotations.xml --coco_categories_config_path example/coco_category_configuration.json
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cvat_xml_file", type=str, required=True)
    parser.add_argument("--coco_categories_config_path", type=str, required=True)

    args = parser.parse_args()
    coco = cvat_image_to_coco(args.cvat_xml_file, args.coco_categories_config_path)
    with open("coco.json", "w") as file:
        json.dump(coco, file)
