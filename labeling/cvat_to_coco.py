from __future__ import annotations
import json
from pathlib import Path
from typing import List
from keypoint_detection.data.coco_parser import CocoImage, CocoInfo, CocoKeypointAnnotation, CocoKeypointCategory, CocoKeypoints
from cvat_keypoints_parser import CVATKeypointsParser, Point, ImageItem

from scripts.xml_to_json import get_dict_from_xml


def cvat_image_to_coco(cvat_xml_path: str, categories: List[CocoKeypointCategory]):

    label_to_category_id_dict = {label: category.id for category in categories for label in category.keypoints}

    cvat_dict = get_dict_from_xml(cvat_xml_path)
    print(cvat_dict)
    cvat_parsed = CVATKeypointsParser(**cvat_dict) 
    print(cvat_parsed.annotations.image[0])
    
    coco_model = CocoKeypoints(images= [], annotations = [], categories = categories)
    annotation_counter = 1
    for cvat_image in cvat_parsed.annotations.image:
        coco_image = CocoImage(file_name=cvat_image.name, height = int(cvat_image.height), width =int(cvat_image.width), id = int(cvat_image.id) + 1)
        coco_model.images.append(coco_image)
        for category in categories:
            category_keypoints = []
            for label in category.keypoints:
                keypoint = get_keypoint_instance_from_cvat_image(cvat_image, label)
                category_keypoints.extend(keypoint)

            coco_model.annotations.append(CocoKeypointAnnotation(category_id=category.id,id=annotation_counter, image_id=coco_image.id,keypoints = category_keypoints))
            annotation_counter += 1
    return coco_model.dict(exclude_none=True)


def get_keypoint_instance_from_cvat_image(cvat_image: ImageItem, label:str) -> List:
    if cvat_image.points is None:
        return [0.0, 0.0, 0]
    if not isinstance(cvat_image.points, list):
        if label == cvat_image.points.label:
            return extract_coco_keypoint_from_cvat_point(cvat_image.points, cvat_image)
        else:
            return [0.0, 0.0, 0]
    for cvat_point in cvat_image.points:
        if label == cvat_point.label:
            return extract_coco_keypoint_from_cvat_point(cvat_point, cvat_image)
    return [0.0, 0.0, 0]
                
def extract_coco_keypoint_from_cvat_point(cvat_point: Point, cvat_image: ImageItem) -> List:
    u = float(cvat_point.points.split(",")[0]) #/ int(cvat_image.width)
    v = float(cvat_point.points.split(",")[1]) #/ int(cvat_image.height)
    f = 1 if cvat_point.occluded == "1" else 2 # occluded = 1 means not visible, which is 1 in COCO; visible in COCO is 2 
    return [u,v,f]

if __name__ == "__main__":
### define your categories here

    onion_category = CocoKeypointCategory(supercategory="tshirt", id = 1, name="tshirt", keypoints = ["neck"], skeleton=[[1,0]])
    categories = [onion_category]
    cvat_xml_path = Path(__file__).parent / "annotations.xml"
    coco = cvat_image_to_coco(cvat_xml_path,categories)
    with open("coco.json", "w") as file:
        json.dump(coco, file)