import unittest

from keypoint_detection.models.metrics import (
    ClassifiedKeypoint,
    DetectedKeypoint,
    Keypoint,
    calculate_ap_from_pr,
    calculate_precision_recall,
    keypoint_classification,
)


class TestKeypointClassification(unittest.TestCase):
    def test_double_match_classification(self):
        gt_keypoints = [Keypoint(10, 10), Keypoint(20, 20), Keypoint(30, 30)]
        det_1_keypoints = [DetectedKeypoint(10, 12, 0.8), DetectedKeypoint(20, 22, 0.3), DetectedKeypoint(20, 23, 0.2)]

        distance = 5

        classified_keypoints = keypoint_classification(det_1_keypoints, gt_keypoints, distance)
        sorted_classified_keypoints = sorted(classified_keypoints, key=lambda x: x.probability, reverse=True)

        self.assertEqual(len(sorted_classified_keypoints), 3)
        self.assertEqual(sorted_classified_keypoints[0].u, 10)
        self.assertEqual(sorted_classified_keypoints[0].v, 12)
        self.assertEqual(sorted_classified_keypoints[0].true_positive, True)
        self.assertEqual(sorted_classified_keypoints[1].true_positive, True)
        self.assertEqual(sorted_classified_keypoints[2].true_positive, False)

    def test_unmatched_match_classification(self):
        gt_keypoints = [Keypoint(10, 10), Keypoint(20, 20), Keypoint(30, 30)]
        det_1_keypoints = [DetectedKeypoint(10, 12, 0.8), DetectedKeypoint(20, 40, 0.3), DetectedKeypoint(20, 50, 0.2)]

        distance = 5

        classified_keypoints = keypoint_classification(det_1_keypoints, gt_keypoints, distance)
        sorted_classified_keypoints = sorted(classified_keypoints, key=lambda x: x.probability, reverse=True)

        self.assertEqual(len(sorted_classified_keypoints), 3)
        self.assertEqual(sorted_classified_keypoints[0].u, 10)
        self.assertEqual(sorted_classified_keypoints[0].v, 12)
        self.assertEqual(sorted_classified_keypoints[0].true_positive, True)
        self.assertEqual(sorted_classified_keypoints[1].true_positive, False)
        self.assertEqual(sorted_classified_keypoints[2].true_positive, False)


class TestMetrics(unittest.TestCase):
    """
    based on example from https://github.com/rafaelpadilla/Object-Detection-Metrics
    """

    def setUp(self):
        self.classified_keypoints = [
            ClassifiedKeypoint(0, 0, 0.95, 0, True),
            ClassifiedKeypoint(0, 0, 0.95, 0, False),
            ClassifiedKeypoint(0, 0, 0.91, 0, True),
            ClassifiedKeypoint(0, 0, 0.88, 0, False),
            ClassifiedKeypoint(0, 0, 0.84, 0, False),
            ClassifiedKeypoint(0, 0, 0.80, 0, False),
            ClassifiedKeypoint(0, 0, 0.78, 0, False),
            ClassifiedKeypoint(0, 0, 0.74, 0, False),
            ClassifiedKeypoint(0, 0, 0.71, 0, False),
            ClassifiedKeypoint(0, 0, 0.70, 0, True),
            ClassifiedKeypoint(0, 0, 0.67, 0, False),
            ClassifiedKeypoint(0, 0, 0.62, 0, True),
            ClassifiedKeypoint(0, 0, 0.54, 0, True),
            ClassifiedKeypoint(0, 0, 0.48, 0, True),
            ClassifiedKeypoint(0, 0, 0.45, 0, False),
            ClassifiedKeypoint(0, 0, 0.45, 0, False),
            ClassifiedKeypoint(0, 0, 0.44, 0, False),
            ClassifiedKeypoint(0, 0, 0.44, 0, False),
            ClassifiedKeypoint(0, 0, 0.43, 0, False),
            ClassifiedKeypoint(0, 0, 0.38, 0, False),
            ClassifiedKeypoint(0, 0, 0.35, 0, False),
            ClassifiedKeypoint(0, 0, 0.23, 0, False),
            ClassifiedKeypoint(0, 0, 0.18, 0, True),
            ClassifiedKeypoint(0, 0, 0.14, 0, False),
        ]
        self.n_gt = 15

    def test_precision_recall(self):
        p, r = calculate_precision_recall(self.classified_keypoints, self.n_gt)

        self.assertEqual(p[0], 1.0)
        self.assertAlmostEqual(p[1], 1.0)
        self.assertAlmostEqual(p[5], 0.4)
        self.assertAlmostEqual(p[10], 0.3)
        self.assertAlmostEqual(p[15], 0.4)
        self.assertAlmostEqual(p[-2], 0.2917, places=3)

        self.assertAlmostEqual(r[0], 0.0, places=3)
        self.assertAlmostEqual(r[1], 0.0666, places=3)
        self.assertAlmostEqual(r[5], 0.1333, places=3)
        self.assertAlmostEqual(r[10], 0.2, places=3)
        self.assertAlmostEqual(r[15], 0.4, places=3)
        self.assertAlmostEqual(r[-2], 0.4666, places=3)

    def test_average_precision(self):
        p, r = calculate_precision_recall(self.classified_keypoints, self.n_gt)
        ap = calculate_ap_from_pr(p, r)
        self.assertAlmostEqual(ap, 0.2456, places=3)
