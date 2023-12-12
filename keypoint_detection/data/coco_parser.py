from typing import List, Optional, Union

from pydantic import BaseModel

"""Custom parser for COCO keypoints JSON"""

LicenseID = int
ImageID = int
CategoryID = int
AnnotationID = int
Segmentation = List[List[Union[float, int]]]
FileName = str
Relativepath = str
Url = str


class CocoInfo(BaseModel):
    description: str
    url: Url
    version: str
    year: int
    contributor: str
    date_created: str


class CocoLicenses(BaseModel):
    url: Url
    id: LicenseID
    name: str


class CocoImage(BaseModel):
    license: Optional[LicenseID] = None
    file_name: Relativepath
    height: int
    width: int
    id: ImageID


class CocoKeypointCategory(BaseModel):
    supercategory: str  # should be set to "name" for root category
    id: CategoryID
    name: str
    keypoints: List[str]
    skeleton: Optional[List[List[int]]] = None


class CocoKeypointAnnotation(BaseModel):
    category_id: CategoryID
    id: AnnotationID
    image_id: ImageID

    num_keypoints: Optional[int] = None
    # COCO keypoints can be floats if they specify the exact location of the keypoint (e.g. from CVAT)
    # even though COCO format specifies zero-indexed integers (i.e. every keypoint in the [0,1]x [0.1] pixel box becomes (0,0)
    keypoints: List[float]

    # TODO: add checks.
    # @validator("keypoints")
    # def check_amount_of_keypoints(cls, v, values, **kwargs):
    #     assert len(v) // 3 == values["num_keypoints"]


class CocoKeypoints(BaseModel):
    """Parser Class for COCO keypoints JSON

    Example:
    with open("path","r") as file:
        data = json.load(file) # dict
        parsed_data = COCOKeypoints(**data)
    """

    info: Optional[CocoInfo] = None
    licenses: Optional[List[CocoLicenses]] = None
    images: List[CocoImage]
    categories: List[CocoKeypointCategory]
    annotations: List[CocoKeypointAnnotation]
