"""Regression cases where a compact rewrite can change the paper protocol."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_paths import DATA_ROOT
sys.path.insert(0, str(DATA_ROOT))
from preprocess import build_stageA_quality_data as quality
from preprocess.build_stageA_select_spk import select_rows
from evaluation.activity import parse, task_prompt
from evaluation.aggregate_mllm import aggregate
from evaluation.alignment import audio_cache_path


class PaperProtocolTests(unittest.TestCase):
    def test_frozen_prompt_bytes(self):
        expected = {'duplex': '96ae966b5e54de30a99a37447450db1abdcf66d87fca1001579b35264c78b495',
                    'semantic': '7b35e13962f988e90ae96d2c90c29af82f0132e4517d4540a2d72c825a8b6a7e'}
        for task, digest in expected.items():
            self.assertEqual(hashlib.sha256(task_prompt(task).encode()).hexdigest(), digest)

    def test_final_speaker_rule_and_energy_boundary(self):
        row = dict(speaker_id='0001', num_windows=100, num_passed=70,
                   pass_rate=.7, mean_score=.9, mean_speech_motion_score=.9, outlier_rate=.01)
        self.assertEqual(select_rows([row]), [row])
        for key, value in [('pass_rate', .699), ('mean_score', .899),
                           ('mean_speech_motion_score', .899), ('outlier_rate', .011)]:
            self.assertEqual(select_rows([dict(row, **{key: value})]), [])
        # Score and hard exclusion intentionally use different boundary conventions.
        self.assertAlmostEqual(quality.motion_energy_score_from_rank(.98), .35)
        self.assertEqual(quality.bucket_from_rank(.98), 'outlier')

    def test_quality_uses_jointwise_displacement_for_flat_poses(self):
        time = np.arange(4, dtype=np.float32)[:, None, None]
        body = np.broadcast_to(time, (4, 21, 3)).copy()
        hands = np.zeros((4, 15, 3), dtype=np.float32)
        poses = {'body_pose': body, 'left_hand_pose': hands, 'right_hand_pose': hands}
        flat = {key: value.reshape(4, -1) for key, value in poses.items()}
        original = quality.compute_window_metrics(poses, 0, 4, 30, [])
        flattened = quality.compute_window_metrics(flat, 0, 4, 30, [])
        self.assertEqual(original, flattened)
        self.assertAlmostEqual(original['body_energy'], np.sqrt(3), places=6)
        self.assertAlmostEqual(original['motion_energy'], .45 * np.sqrt(3), places=6)

    def test_audio_cache_tracks_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / 'audio.wav'
            wav.write_bytes(b'first waveform')
            first = audio_cache_path(root, wav)
            wav.write_bytes(b'replaced waveform')
            self.assertNotEqual(audio_cache_path(root, wav), first)

    def test_four_judgments_are_penalized_before_averaging(self):
        records = []
        values = [1., 0., 1., 0.]
        for index, (provider, repeat) in enumerate((p, r) for p in ['gemini', 'seed'] for r in [1, 2]):
            response = {'Timing': values[index], 'Appropriateness': values[index],
                        'Gesture backchannel visibility': values[index], 'Silence motion penalty': 0,
                        'Overactivity penalty': 0, 'Silence evidence': [], 'Overactivity evidence': []}
            records.append(dict(valid=True, group_id='clip', task='duplex', provider=provider,
                                round=repeat, response=json.dumps(response), scores={'final': values[index]}))
        entry = {'group_id': 'clip', 'task': 'duplex'}
        self.assertEqual(aggregate(records, [entry])['summary']['duplex']['mean'], .5)
        with self.assertRaisesRegex(AssertionError, 'Incomplete'):
            aggregate(records[:-1], [entry])
        with self.assertRaisesRegex(AssertionError, 'Duplicate'):
            aggregate(records + records[:1], [entry])
        for record in records:
            record['signature'] = {'hashes': {'candidate': 'same', 'reference': 'same'}, 'prompt_sha256': 'same'}
        records[-1]['signature']['prompt_sha256'] = 'changed'
        with self.assertRaisesRegex(AssertionError, 'Mixed judge inputs'):
            aggregate(records, [entry])

    def test_invalid_judge_history_survives_retry_and_zero_score_is_cached(self):
        from evaluation.score_mllm import score_one
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'candidate.mp4'
            reference = root / 'reference.mp4'
            video.write_bytes(b'candidate')
            reference.write_bytes(b'reference')
            entry = dict(group_id='clip', task='duplex', video=str(video), quiet_windows=[],
                         metadata={'reference_video': str(reference)})
            args = SimpleNamespace(output_dir=root, provider='gemini', model='test',
                                   base_url='https://example.invalid', azure_endpoint=None,
                                   video_format='video_url', api_version='2024-03-01-preview', retries=1)
            zero = {'Timing': 0, 'Appropriateness': 0, 'Gesture backchannel visibility': 0,
                    'Silence motion penalty': 0, 'Overactivity penalty': 0,
                    'Silence evidence': [], 'Overactivity evidence': []}
            placeholder = ('reference time/start-change-end; candidate same-time start-change-end; '
                           'strongest local evidence; comparison judgment')
            payloads = iter(['invalid JSON', json.dumps(dict(zero, Analysis=placeholder)),
                             json.dumps(dict(zero, Analysis='No visible response during the rated interval.'))])
            calls = []
            def create(**kwargs):
                calls.append(kwargs)
                self.assertEqual(kwargs['max_tokens'], 30000)
                self.assertEqual(kwargs['temperature'], 0)
                self.assertNotIn('seed', kwargs)
                return SimpleNamespace(model='returned-test-model', choices=[
                    SimpleNamespace(message=SimpleNamespace(content=next(payloads)))])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    score_one(entry, 1, client, args)
            result = score_one(entry, 1, client, args)
            self.assertEqual(result['scores']['final'], 0)
            self.assertEqual(len(result['attempts']), 2)
            self.assertEqual(result['returned_model'], 'returned-test-model')
            self.assertEqual(score_one(entry, 1, client, args)['scores']['final'], 0)
            self.assertEqual(len(calls), 3)

    def test_face_metrics_preserve_sample_std_and_repeated_vertices(self):
        import torch
        from evaluation import face_metrics
        pred = torch.tensor([[[[0., 0., 0.], [0., 0., 0.]],
                              [[.001, .001, 0.], [.003, 0., 0.]]]])
        gt = torch.zeros_like(pred)
        with patch.object(face_metrics, 'lip_indices', [0, 1]), \
             patch.object(face_metrics, 'upper_indices', [0, 0, 1]):
            lve, mhd, fdd = face_metrics.calc_val_metrics(pred, gt)
        self.assertAlmostEqual(float(lve), np.sqrt(9 / 2), places=6)
        self.assertAlmostEqual(float(mhd), np.sqrt(11 / 4), places=6)
        self.assertAlmostEqual(float(fdd), 100 * (2 + 2 + 3) / (3 * np.sqrt(2)), places=4)

    def test_judge_requires_numeric_timestamps(self):
        response = {'Timing': 1, 'Appropriateness': 1, 'Gesture backchannel visibility': 1,
                    'Silence motion penalty': 0, 'Overactivity penalty': .2, 'Silence evidence': [],
                    'Overactivity evidence': [{'start_sec': True, 'end_sec': 2,
                        'candidate_motion': 'nod', 'reference_motion': 'still', 'why_unjustified': 'unrelated'}]}
        with self.assertRaises(AssertionError):
            parse(json.dumps(response), {'task': 'duplex', 'quiet_windows': []})


if __name__ == '__main__':
    unittest.main()
