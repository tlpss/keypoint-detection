"""Parser for configuration of COCO categories to convert CVAT XML to COCO Keypoints
"""
from typing import List

from pydantic import BaseModel


class COCOSemanticTypeConfig(BaseModel):
    name: str
    n_keypoints: int


class COCOCategoryConfig(BaseModel):
    supercategory: str
    id: int
    name: str
    semantic_types: List[COCOSemanticTypeConfig]


class COCOCategoriesConfig(BaseModel):
    categories: List[COCOCategoryConfig]
