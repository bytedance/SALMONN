"""Preserve original facial crop interpolation and full-sequence crop context."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data'))
from preprocess.extract_kaolrm import face_crops, prepare


def keypoints(count):
    result = np.zeros((count, 137, 3), np.float32)
    result[:, 23:91, 0] = np.arange(count)[:, None] * 2 + np.linspace(30, 70, 68)
    result[:, 23:91, 1] = np.linspace(40, 80, 68)
    result[:, 23:91, 2] = 1
    return result


class FlameExtractionTests(unittest.TestCase):
    def test_missing_measurements_keep_interpolated_crops_on_valid_body_frames(self):
        points = keypoints(80)
        points[20:40, 23:91, 2] = 0
        valid = np.ones(80, bool)
        valid[30] = False
        boxes, detected = face_crops(points, valid, 640, 480)
        self.assertFalse(detected[25])
        self.assertGreaterEqual(boxes[25, 0], 0)
        np.testing.assert_array_equal(boxes[30], [-1, -1, -1, -1])

    def test_frame_cap_keeps_the_full_extraction_crop_prefix(self):
        import cv2
        capture = unittest.mock.Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda field: {
            cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_COUNT: 67,
            cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480}[field]
        with tempfile.TemporaryDirectory() as directory:
            pose = Path(directory) / 'pose.npz'
            np.savez(pose, **{'boxes_and_keypoints:keypoints': keypoints(80),
                             'smplh:is_valid': np.ones(80, bool)})
            with patch('preprocess.extract_kaolrm.cv2.VideoCapture', return_value=capture):
                full = prepare('video.mp4', pose)
                capped = prepare('video.mp4', pose, 24)
            self.assertEqual(int(capped['source_pose_num_frames']), 80)
            for key in ('crop_boxes', 'frame_indices', 'pose_indices', 'face_detected'):
                np.testing.assert_array_equal(capped[key], full[key][:24])


if __name__ == '__main__':
    unittest.main()
