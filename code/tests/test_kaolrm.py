"""Crop validity, timeline alignment and raw FLAME-to-training interface regressions."""
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_paths import DATA_ROOT
sys.path.insert(0, str(DATA_ROOT))
if cv2 is not None:
    from preprocess.extract_kaolrm import extract, face_crops


def landmarks(frames):
    points = np.zeros((frames, 133, 3), np.float32)
    points[:, 23:91, 0] = np.linspace(20, 44, 68)
    points[:, 23:91, 1] = np.linspace(22, 42, 68)
    points[:, 23:91, 2] = 1
    return points


@unittest.skipIf(cv2 is None, 'Install the optional KaoLRM extraction requirements for these tests')
class KaoLRMTests(unittest.TestCase):
    def test_crop_interpolation_keeps_valid_body_frames(self):
        points = landmarks(40)
        valid = np.ones(40, bool)
        points[12:18, 23:91, 2] = 0
        valid[25] = False
        boxes, detected = face_crops(points, valid, 64, 64)
        expected = valid.copy()
        expected[12:18] = False
        np.testing.assert_array_equal(detected, expected)
        np.testing.assert_array_equal(boxes[~valid], -1)
        self.assertTrue((boxes[12:18] >= 0).all())
        self.assertTrue((boxes[expected, :2] >= 0).all())
        self.assertTrue((boxes[expected, 2:] <= 64).all())

    def test_no_detection_does_not_create_zero_supervision(self):
        boxes, detected = face_crops(np.zeros((10, 133, 3)), np.ones(10, bool), 64, 64)
        self.assertFalse(detected.any())
        self.assertTrue((boxes == -1).all())

    def test_real_video_timeline_raw_parameters_and_training_layout(self):
        from dataset import Gesture_Dataset
        class Predictor:
            count = 0
            def predict(self, crop):
                value = self.count
                self.count += 1
                return {key: np.full(size, value, np.float32)
                        for key, size in [('expression', 50), ('pose', 6), ('shape', 100)]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'source.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 25., (64, 64))
            self.assertTrue(writer.isOpened())
            for index in range(10):
                writer.write(np.full((64, 64, 3), index * 20, np.uint8))
            writer.release()
            points, valid = landmarks(12), np.ones(12, bool)
            points[4:7, 23:91, 2] = 0
            valid[9] = False
            pose = root / 'source.npz'
            np.savez(pose, **{'boxes_and_keypoints:keypoints': points, 'smplh:is_valid': valid})
            output = root / 'raw.npz'
            info = extract(dict(pose_file=str(pose), video_file=str(video)), output, Predictor())
            self.assertEqual(info['valid_frames'], 11)
            with np.load(output) as data:
                expected = valid.copy()
                np.testing.assert_array_equal(data['valid_mask'], expected)
                np.testing.assert_array_equal(data['frame_indices'], np.clip(np.rint(np.arange(12) / 30 * 25), 0, 9))
                np.testing.assert_array_equal(data['expression50'][expected, 0], np.arange(11))
                np.testing.assert_array_equal(data['motioncode'][~expected], 0)
                np.testing.assert_array_equal(data['expression'][:, 50:], 0)
                self.assertEqual(data['motioncode'].shape, (12, 112))
                training = Gesture_Dataset._select_unils_motioncode(data['motioncode'])
                self.assertEqual(training.shape, (12, 108))
                np.testing.assert_array_equal(training[:, :104], data['motioncode'][:, :104])
                np.testing.assert_array_equal(training[:, 104:], 0)
                self.assertNotIn('source_video', data.files)
                self.assertEqual(str(data['param_smoothing']), 'none')
                self.assertTrue(all(data[key].dtype != object for key in data.files))


if __name__ == '__main__':
    unittest.main()
