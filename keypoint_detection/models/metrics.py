"""
Implementation of (mean) Average Precision metric for 2D keypoint detection.
"""

from __future__ import annotations  # allow typing of own class objects

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
from torchmetrics import Metric


@dataclass
class Keypoint:
    """A simple class datastructure for Keypoints,
    dataclass is chosen over named tuple because this class is inherited by other classes
    """

    u: int
    v: int

    def l2_distance(self, keypoint: Keypoint):
        return math.sqrt((self.u - keypoint.u) ** 2 + (self.v - keypoint.v) ** 2)


@dataclass
class DetectedKeypoint(Keypoint):
    probability: float


@dataclass(unsafe_hash=True)
class ClassifiedKeypoint(DetectedKeypoint):
    """
    DataClass for a classified keypoint, where classified means determining if the detection is a True Positive of False positive,
     with the given treshold distance and the gt keypoints from the frame

    a hash is required for torch metric
    cf https://github.com/PyTorchLightning/metrics/blob/2c8e46f87cb67186bff2c7b94bf1ec37486873d4/torchmetrics/metric.py#L570
    unsafe_hash -> dirty fix to allow for hash w/o explictly telling python the object is immutable.
    """

    threshold_distance: float
    true_positive: bool


def keypoint_classification(
    detected_keypoints: List[DetectedKeypoint],
    ground_truth_keypoints: List[Keypoint],
    threshold_distance: float,
) -> List[ClassifiedKeypoint]:
    """Classifies keypoints of a **single** frame in True Positives or False Positives by searching for unused gt keypoints in prediction probability order
    that are within distance d of the detected keypoint.

    Args:
        detected_keypoints (List[DetectedKeypoint]): The detected keypoints in the frame
        ground_truth_keypoints (List[Keypoint]): The ground truth keypoints of a frame
        threshold_distance: maximal distance in pixel coordinate space between detected keypoint and ground truth keypoint to be considered a TP

    Returns:
        List[ClassifiedKeypoint]: Keypoints with TP label.
    """
    classified_keypoints: List[ClassifiedKeypoint] = []

    ground_truth_keypoints = copy.deepcopy(
        ground_truth_keypoints
    )  # make deep copy to do local removals (pass-by-reference..)

    for detected_keypoint in sorted(detected_keypoints, key=lambda x: x.probability, reverse=True):
        matched = False
        for gt_keypoint in ground_truth_keypoints:
            distance = detected_keypoint.l2_distance(gt_keypoint)
            if distance < threshold_distance:
                classified_keypoint = ClassifiedKeypoint(
                    detected_keypoint.u,
                    detected_keypoint.v,
                    detected_keypoint.probability,
                    threshold_distance,
                    True,
                )
                matched = True
                # remove keypoint from gt to avoid muliple matching
                ground_truth_keypoints.remove(gt_keypoint)
                break
        if not matched:
            classified_keypoint = ClassifiedKeypoint(
                detected_keypoint.u,
                detected_keypoint.v,
                detected_keypoint.probability,
                threshold_distance,
                False,
            )
        classified_keypoints.append(classified_keypoint)

    return classified_keypoints


def calculate_precision_recall(
    classified_keypoints: List[ClassifiedKeypoint], total_ground_truth_keypoints: int
) -> Tuple[List[float], List[float]]:
    """Calculates precision recall points on the curve for the given keypoints by varying the treshold probability to all detected keypoints
     (i.e. by always taking one additional keypoint als a predicted event)

    Note that this function is tailored towards a Detector, not a Classifier. For classifiers, the outputs contain both TP, FP and FN. Whereas for a Detector the
    outputs only define the TP and the FP; the FN are not contained in the output as the point is exactly that the detector did not detect this event.

    A detector is a ROI finder + classifier and the ROI finder could miss certain regions, which results in FNs that are hence never passed to the classifier.

    This also explains why the scikit average_precision function states it is for Classification tasks only. Since it takes "total_gt_events" to be the # of positive_class labels.
    The function can however be used by using as label (TP = 1, FP = 0) and by then multiplying the result with TP/(TP + FN) since the recall values are then corrected
    to take the unseen events (FN's) into account as well. They do not matter for precision calcultations.
    Args:
        classified_keypoints (List[ClassifiedKeypoint]):
        total_ground_truth_keypoints (int):

    Returns:
        Tuple[List[float], List[float]]: precision, recall entries. First entry is (1,0); last entry is (0,1).
    """
    precision = [1.0]
    recall = [0.0]

    true_positives = 0
    false_positives = 0

    for keypoint in sorted(classified_keypoints, key=lambda x: x.probability, reverse=True):
        if keypoint.true_positive:
            true_positives += 1
        else:
            false_positives += 1

        precision.append(true_positives / (true_positives + false_positives))
        recall.append(true_positives / total_ground_truth_keypoints)

    precision.append(0.0)
    recall.append(1.0)
    return precision, recall


def calculate_ap_from_pr(precision: List[float], recall: List[float]) -> float:
    """Calculates the Average Precision using the AUC definition (COCO-style)

    # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    # AUC AP.

    Args:
        precision (List[float]):
        recall (List[float]):

    Returns:
        (float): average precision (between 0 and 1)
    """

    smoothened_precision = copy.deepcopy(precision)

    for i in range(len(smoothened_precision) - 2, 0, -1):
        smoothened_precision[i] = max(smoothened_precision[i], smoothened_precision[i + 1])

    ap = 0
    for i in range(len(recall) - 1):
        ap += (recall[i + 1] - recall[i]) * smoothened_precision[i + 1]

    return ap


class KeypointAPMetric(Metric):
    """torchmetrics-like interface for the Average Precision implementation"""

    def __init__(self, keypoint_threshold_distance: float, dist_sync_on_step=False):
        """

        Args:
            keypoint_threshold_distance (float): distance from ground_truth keypoint that is used to classify keypoint as TP or FP.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.keypoint_threshold_distance = keypoint_threshold_distance

        default: Callable = lambda: []
        self.add_state("classified_keypoints", default=default(), dist_reduce_fx="cat")  # list of ClassifiedKeypoints
        self.add_state("total_ground_truth_keypoints", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, detected_keypoints: List[DetectedKeypoint], gt_keypoints: List[Keypoint]):

        classified_img_keypoints = keypoint_classification(
            detected_keypoints, gt_keypoints, self.keypoint_threshold_distance
        )

        self.classified_keypoints += classified_img_keypoints

        self.total_ground_truth_keypoints += len(gt_keypoints)

    def compute(self):
        p, r = calculate_precision_recall(self.classified_keypoints, int(self.total_ground_truth_keypoints.cpu()))
        m_ap = calculate_ap_from_pr(p, r)
        return m_ap


class KeypointAPMetrics(Metric):
    """
    Torchmetrics-like interface for calculating average precisions over different keypoint_threshold_distances.
    Uses KeypointAPMetric class.
    """

    def __init__(self, keypoint_threshold_distances: List[float], dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.ap_metrics = [KeypointAPMetric(dst, dist_sync_on_step) for dst in keypoint_threshold_distances]

    def update(self, detected_keypoints: List[DetectedKeypoint], gt_keypoints: List[Keypoint]):
        for metric in self.ap_metrics:
            metric.update(detected_keypoints, gt_keypoints)

    def compute(self) -> Dict[float, float]:
        result_dict = {}
        for metric in self.ap_metrics:
            result_dict.update({metric.keypoint_threshold_distance: metric.compute()})
        return result_dict

    def reset(self) -> None:
        for metric in self.ap_metrics:
            metric.reset()
