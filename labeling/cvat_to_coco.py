from __future__ import annotations
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
from typing import List
from keypoint_detection.data.coco_parser import CocoImage, CocoInfo, CocoKeypointAnnotation, CocoKeypointCategory, CocoKeypoints
from cvat_keypoints_parser import CVATKeypointsParser, Point, ImageItem
from pydantic import BaseModel
from scripts.xml_to_json import get_dict_from_xml


def cvat_image_to_coco(cvat_xml_path: str, category_dict):


    cvat_dict = get_dict_from_xml(cvat_xml_path)
    cvat_parsed = CVATKeypointsParser(**cvat_dict) 
    
    coco_model = CocoKeypoints(images= [], annotations = [], categories = [])
    annotation_counter = 1


    for category_id, category in enumerate(category_dict["categories"]):
        category_keypoint_names = get_coco_keypoint_names_from_category_config(category)
        coco_model.categories.append(CocoKeypointCategory(id=category_id, name = category["name"],supercategory=category["supercategory"],keypoints = category_keypoint_names))
        
        category_name = category["name"]
        for cvat_image in cvat_parsed.annotations.image:
            coco_image = CocoImage(file_name=cvat_image.name, height = int(cvat_image.height), width =int(cvat_image.width), id = int(cvat_image.id) + 1)
            coco_model.images.append(coco_image)
            n_image_category_instances = get_n_category_instance_in_image(cvat_image, category_name)
            for instance_id in range(1, n_image_category_instances + 1): # IDs start with 1
                instance_category_keypoints = []
                for semantic_type in category["semantic_types"]:
                    semantic_type_name = semantic_type["name"]
                    n_keypoints = semantic_type["n_keypoints"]
                    keypoints = get_semantic_type_keypoints_from_instance_in_cvat_image(cvat_image, semantic_type_name, instance_id)

                    keypoints.extend([0.0] * (3 * n_keypoints - len(keypoints)))
                    instance_category_keypoints.extend(keypoints)

                coco_model.annotations.append(CocoKeypointAnnotation(category_id=category_id,id=annotation_counter, image_id=coco_image.id,keypoints = instance_category_keypoints))
                annotation_counter += 1
    return coco_model.dict(exclude_none=True)

def get_n_category_instance_in_image(cvat_image: ImageItem, category_name: str) -> int: 
    if cvat_image.points is None:
        return 0
    if not isinstance(cvat_image.points, list):
        if get_category_from_cvat_label(cvat_image.points.label)== category_name:
            return int(cvat_image.points.group_id)
        else:
            return 0
    max_group_id = 1
    for cvat_point in cvat_image.points:
        if get_category_from_cvat_label(cvat_point.label)== category_name:
            max_group_id = max(max_group_id, int(cvat_point.group_id))
    return max_group_id 

def get_category_from_cvat_label(label:str):
    split = label.split(".") 
    assert len(split) == 2, " label was not formatted as category.semantic_type"
    return label.split(".")[0]
def get_semantic_type_from_cvat_label(label:str): 
    return label.split(".")[1]
def get_coco_keypoint_names_from_category_config(config: dict):
    #TODO: append index for > 1 
    return [""]

def get_semantic_type_keypoints_from_instance_in_cvat_image(cvat_image: ImageItem, semantic_type:str, instance_id:int) -> List:
    instance_id = str(instance_id)
    if cvat_image.points is None:
        return [0.0, 0.0, 0]
    if not isinstance(cvat_image.points, list):
        if semantic_type == get_semantic_type_from_cvat_label(cvat_image.points.label) and instance_id == cvat_image.points.group_id:
            return extract_coco_keypoint_from_cvat_point(cvat_image.points, cvat_image)
        else:
            return [0.0, 0.0, 0]
    keypoints = []
    for cvat_point in cvat_image.points:
        if semantic_type == get_semantic_type_from_cvat_label(cvat_point.label) and instance_id == cvat_point.group_id:
            keypoints.extend(extract_coco_keypoint_from_cvat_point(cvat_point, cvat_image))
    return keypoints
                
def extract_coco_keypoint_from_cvat_point(cvat_point: Point, cvat_image: ImageItem) -> List:
    u = float(cvat_point.points.split(",")[0]) #/ int(cvat_image.width)
    v = float(cvat_point.points.split(",")[1]) #/ int(cvat_image.height)
    f = 1 if cvat_point.occluded == "1" else 2 # occluded = 1 means not visible, which is 1 in COCO; visible in COCO is 2 
    return [u,v,f]



class COCOSemanticTypeConfig(BaseModel):
    name: str
    n_keypoints: int
class COCOCategoryConfig(BaseModel):
    name: str
    semantic_types: List[COCOSemanticTypeConfig]


class COCOCategoriesConfig(BaseModel):
    categories: List[COCOCategoryConfig]

if __name__ == "__main__":
### define your categories here

    category_dict = { 
        "categories":  
        [
            {
                "name" : "tshirt",
                "supercategory": "cloth",
                "semantic_types" : [
                    {
                        "name": "neck",
                        "n_keypoints" : 1
                    },
                    {
                        "name": "shoulder",
                        "n_keypoints" : 2
                    },

                ]
            }
        ]
    }
    parsed_categor_config = COCOCategoriesConfig(**category_dict)

    cvat_xml_path = Path(__file__).parent / "annotations.xml"
    coco = cvat_image_to_coco(cvat_xml_path,category_dict)
    with open("coco.json", "w") as file:
        json.dump(coco, file)