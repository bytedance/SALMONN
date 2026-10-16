"""Verify provider defaults, overlapping requests and independently bounded queues."""
import concurrent.futures
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import evaluate, score_mllm


class ParallelMLLMTests(unittest.TestCase):
    def test_video_format_defaults_and_explicit_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            args = evaluate.parse_args(['--output-dir', 'out'])
            self.assertEqual((args.gemini_video_format, args.seed_video_format), ('azure_image', 'video_url'))
            args = evaluate.parse_args(['--output-dir', 'out', '--gemini-video-format', 'video_url'])
            self.assertEqual(args.gemini_video_format, 'video_url')
            for provider, default in [('gemini', 'azure_image'), ('seed', 'video_url')]:
                common = ['--provider', provider, '--model', 'model', '--video-dir', 'videos', '--output-dir', 'out']
                self.assertEqual(score_mllm.parse_args(common).video_format, default)
                self.assertEqual(score_mllm.parse_args(common + ['--video-format', 'video_url']).video_format, 'video_url')
        with patch.dict(os.environ, {'GEMINI_VIDEO_FORMAT': 'video_url'}, clear=True):
            self.assertEqual(evaluate.parse_args(['--output-dir', 'out']).gemini_video_format, 'video_url')

    def test_provider_limits_override_the_fallback_independently(self):
        options = ['--output-dir', 'out', '--video-dir', 'videos', '--max-pending', '1',
                   '--gemini-base-url', 'https://gemini.invalid/v1', '--gemini-api-key', 'test-gemini',
                   '--seed-base-url', 'https://seed.invalid/v1', '--seed-model', 'seed-model', '--seed-api-key', 'test-seed']
        with patch.dict(os.environ, {}, clear=True):
            args = evaluate.parse_args(options)
            self.assertEqual(evaluate.provider_options(args, 'gemini', Path('out'), ['clip']).max_pending, 1)
            self.assertEqual(evaluate.provider_options(args, 'seed', Path('out'), ['clip']).max_pending, 1)
            args = evaluate.parse_args(options + ['--gemini-max-pending', '2', '--seed-max-pending', '6'])
            self.assertEqual(evaluate.provider_options(args, 'gemini', Path('out'), ['clip']).max_pending, 2)
            self.assertEqual(evaluate.provider_options(args, 'seed', Path('out'), ['clip']).max_pending, 6)
            args.seed_max_pending = 0
            with self.assertRaisesRegex(AssertionError, 'positive'):
                evaluate.provider_options(args, 'seed', Path('out'), ['clip'])

    def test_provider_pools_overlap_and_bound_all_submitted_jobs(self):
        real_executor = concurrent.futures.ThreadPoolExecutor
        for gemini_limit, seed_limit in [(1, 1), (2, 3), (4, 4)]:
            with self.subTest(gemini=gemini_limit, seed=seed_limit), tempfile.TemporaryDirectory() as directory:
                limits = {'gemini': gemini_limit, 'seed': seed_limit}
                args = [SimpleNamespace(provider=provider, model=provider, rounds=2,
                                        max_pending=limit, output_dir=Path(directory) / provider)
                        for provider, limit in limits.items()]
                lock = threading.Lock()
                barrier = threading.Barrier(sum(limits.values()))
                submitted = {provider: [] for provider in limits}
                peak = {provider: 0 for provider in limits}
                calls = []
                started = {provider: 0 for provider in limits}
                active = {provider: 0 for provider in limits}
                total_active_peak = 0

                def prepare(options):
                    options.output_dir.mkdir()
                    return SimpleNamespace(provider=options.provider), [{'group_id': f'clip_{i}'} for i in range(4)]

                def score(entry, repeat, client, options):
                    nonlocal total_active_peak
                    self.assertEqual(client.provider, options.provider)
                    with lock:
                        calls.append((options.provider, entry['group_id'], repeat))
                        started[options.provider] += 1
                        first_batch = started[options.provider] <= limits[options.provider]
                        active[options.provider] += 1
                        total_active_peak = max(total_active_peak, sum(active.values()))
                    try:
                        if first_batch:
                            # Both services must start; a global cap would block the 5/8-job cases.
                            barrier.wait(timeout=10)
                        time.sleep(.01)
                    finally:
                        with lock:
                            active[options.provider] -= 1

                class TrackingExecutor(real_executor):
                    def submit(self, function, *arguments, **kwargs):
                        future = super().submit(function, *arguments, **kwargs)
                        if function is score:
                            provider = arguments[-1].provider
                            with lock:
                                submitted[provider].append(future)
                                peak[provider] = max(peak[provider], sum(not item.done() for item in submitted[provider]))
                        return future

                with patch.object(score_mllm, 'prepare_provider', side_effect=prepare), \
                     patch.object(score_mllm, 'score_one', new=score), \
                     patch.object(score_mllm.futures, 'ThreadPoolExecutor', TrackingExecutor), \
                     contextlib.redirect_stdout(io.StringIO()):
                    score_mllm.run_many(args)
                self.assertEqual(peak, limits)
                self.assertEqual(total_active_peak, sum(limits.values()))
                self.assertEqual(len(calls), 16)
                self.assertEqual(len(set(calls)), 16)
                for options in args:
                    summary = json.loads((options.output_dir / f'{options.provider}_summary.json').read_text())
                    self.assertEqual(summary['valid'], 8)


if __name__ == '__main__':
    unittest.main()
