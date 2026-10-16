"""CPU regressions for release metadata, cache identity and pooled metrics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_release
from evaluation.pose_fid import moments, frechet_distance
from evaluation.activity import parse


class ReleaseTests(unittest.TestCase):
    def test_private_npz_metadata(self):
        # NPY Unicode/bytes/object encodings must all expose private paths.
        private = '/' + 'mnt/' + 'bn/' + 'private/example.mp4'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(validate_release, 'ROOT', root):
                for dtype in ['U', 'S', object]:
                    path = root / 'metadata.npz'
                    np.savez(path, source_video=np.array(private, dtype=dtype))
                    with self.assertRaises(AssertionError):
                        validate_release.check_private_metadata(path)
                np.savez(path, source_video='testset/reference_videos/example.mp4')
                validate_release.check_private_metadata(path)

    def test_pose_fid_pools_frames(self):
        rng = np.random.default_rng(42)
        pred = [rng.normal(size=(8, 3)), rng.normal(size=(19, 3)) + 2]
        gt = [rng.normal(size=(8, 3)), rng.normal(size=(19, 3))]
        pooled = frechet_distance(list(map(moments, pred)), list(map(moments, gt)))
        concatenated = frechet_distance([moments(np.concatenate(pred))], [moments(np.concatenate(gt))])
        self.assertAlmostEqual(pooled, concatenated, places=12)
        self.assertLess(frechet_distance([moments(gt[0])], [moments(gt[0])]), 1e-10)

    def test_cache_rejects_reordered_sources(self):
        from dataset import Gesture_Dataset
        dataset = Gesture_Dataset.__new__(Gesture_Dataset)
        dataset.has_gesture_target = True
        dataset.has_expression_target = False
        dataset.total_num_motion_codebooks = 18
        dataset.data = [{'pose_file': 'first.npz'}]
        with tempfile.TemporaryDirectory() as directory:
            dataset.motion_token_cache_dir = directory
            np.savez(Path(directory) / '000000.npz',
                     sample_key=dataset.motion_cache_sample_key(0),
                     start_frames=[0], motion_codes=np.zeros((1, 2, 18), dtype=np.int16))
            dataset._load_precomputed_motion_codes(0)
            dataset.data = [{'pose_file': 'second.npz'}]
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                dataset._load_precomputed_motion_codes(0)

    def test_judge_rejects_duplicate_interval_penalty(self):
        evidence = {'start_sec': 1, 'end_sec': 2, 'candidate_motion': 'nod',
                    'reference_motion': 'still', 'why_unjustified': 'no response needed',
                    'target_speech_absent': True}
        response = {'Timing': 1, 'Appropriateness': 1, 'Gesture backchannel visibility': 1,
                    'Silence motion penalty': .2, 'Overactivity penalty': .3,
                    'Silence evidence': [evidence], 'Overactivity evidence': [evidence]}
        with self.assertRaisesRegex(AssertionError, 'duplicate interval'):
            parse(json.dumps(response), {'task': 'duplex', 'quiet_windows': [[1, 3]]})

    def test_output_directory_equals_syntax(self):
        import infer
        with tempfile.TemporaryDirectory() as directory:
            jobs = Path(directory) / 'jobs.json'
            with patch.object(sys, 'argv', ['infer.py', '--test-id', 'fro_00038', '--output-dir=' + directory]), \
                 patch('scripts.prepare_testset.prepare', return_value=jobs) as prepare, \
                 patch.object(infer.subprocess, 'run') as run:
                infer.main()
            self.assertEqual(prepare.call_args.args[0], Path(directory))
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--audio-json') + 1], str(jobs))


if __name__ == '__main__':
    unittest.main()
