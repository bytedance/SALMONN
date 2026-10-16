"""Check public model selection and independent evaluation/credential routing."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import infer
from evaluation import evaluate, score_mllm


class EvaluationGroupTests(unittest.TestCase):
    def test_explicit_model_path_resolves_before_subprocess_changes_directory(self):
        for flag in ['--model-path', '--model_path']:
            with patch.object(sys, 'argv', ['infer.py', flag, 'my_model', '--pose-guide', 'guide.npz']), \
                 patch.object(infer.subprocess, 'run') as run:
                infer.main()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--model-path') + 1], str(Path('my_model').resolve()))
            self.assertEqual(run.call_args.kwargs['cwd'], ROOT)
            self.assertIn('--pose-guide', command)

    def test_automatic_guided_checkpoint_selection_is_preserved(self):
        with patch.object(sys, 'argv', ['infer.py', '--pose-guide', 'guide.npz']), \
             patch.object(infer.subprocess, 'run') as run:
            infer.main()
        self.assertNotIn('--model-path', run.call_args.args[0])

    def test_metric_groups_require_only_their_assets(self):
        examples = [(['--prediction-dir', 'pred', '--smplx-model', 'skeleton'], 'motion'),
                    (['--metrics', 'face', '--prediction-dir', 'pred', '--flame-asset-dir', 'flame'], 'face'),
                    (['--metrics', 'mllm', '--mllm-score-dir', 'scores'], 'mllm')]
        for options, expected in examples:
            args = evaluate.parse_args(['--output-dir', 'out', *options])
            self.assertEqual(evaluate.metric_groups(args), [expected])
        args = evaluate.parse_args(['--metrics', 'face', '--prediction-dir', 'pred',
                                    '--flame-asset-dir', 'flame', '--output-dir', 'out', '--pose-diagnostics'])
        with self.assertRaisesRegex(AssertionError, 'requires motion'):
            evaluate.metric_groups(args)

    def test_provider_settings_are_separate_and_pending_jobs_are_bounded(self):
        args = evaluate.parse_args(['--output-dir', 'out', '--video-dir', 'videos',
                                    '--gemini-api-key', 'gemini-test', '--gemini-base-url', 'https://gemini.invalid/v1',
                                    '--seed-api-key', 'seed-test', '--seed-base-url', 'https://seed.invalid/v1',
                                    '--seed-model', 'seed-deployment'])
        with patch.dict(os.environ, {}, clear=True):
            gemini = evaluate.provider_options(args, 'gemini', Path('out'), ['clip'])
            seed = evaluate.provider_options(args, 'seed', Path('out'), ['clip'])
        self.assertEqual((gemini.api_key, seed.api_key), ('gemini-test', 'seed-test'))
        self.assertEqual((gemini.base_url, seed.base_url), ('https://gemini.invalid/v1', 'https://seed.invalid/v1'))
        self.assertEqual((gemini.rounds, seed.rounds), (2, 2))
        args.max_pending = 5
        with self.assertRaisesRegex(AssertionError, 'four submitted Gemini'):
            evaluate.provider_options(args, 'gemini', Path('out'), ['clip'])

    def test_explicit_key_override_and_environment_key_reach_the_correct_client(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / 'testset').mkdir()
            (data / 'testset/judge_context.json').write_text('[]')
            for azure, direct, expected in [(False, 'explicit-test-key', 'explicit-test-key'),
                                            (True, None, 'environment-test-key')]:
                endpoint = ['--azure-endpoint', 'https://azure.invalid'] if azure else ['--base-url', 'https://base.invalid/v1']
                options = ['--api-key', direct] if direct else []
                args = score_mllm.parse_args(['--video-dir', directory, '--output-dir', str(data / 'out'),
                                             '--provider', 'gemini', '--model', 'test-model',
                                             '--api-key-env', 'TEST_EVALUATION_KEY', *endpoint, *options])
                with patch.dict(os.environ, {'TEST_EVALUATION_KEY': 'environment-test-key'}, clear=True), \
                     patch.object(score_mllm, 'DATA_ROOT', data), \
                     patch('openai.OpenAI') as regular, patch('openai.AzureOpenAI') as azure_client:
                    score_mllm.run(args)
                chosen, other = (azure_client, regular) if azure else (regular, azure_client)
                self.assertEqual(chosen.call_args.kwargs['api_key'], expected)
                self.assertEqual(chosen.call_args.kwargs['max_retries'], 0)
                other.assert_not_called()
                self.assertNotIn(expected, (data / 'out/gemini_summary.json').read_text())

    def test_cached_mllm_needs_neither_objective_assets_nor_api_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / 'testset').mkdir()
            (data / 'testset/manifest.jsonl').write_text(json.dumps({'group_id': 'clip', 'human_approved_task': 'duplex'}) + '\n')
            (data / 'testset/judge_context.json').write_text(json.dumps([{'group_id': 'clip', 'task': 'duplex'}]))
            scores = data / 'scores'
            scores.mkdir()
            response = json.dumps({'Timing': 1, 'Appropriateness': 1, 'Gesture backchannel visibility': 1,
                                   'Silence motion penalty': 0, 'Overactivity penalty': 0,
                                   'Silence evidence': [], 'Overactivity evidence': []})
            for provider in ['gemini', 'seed']:
                for repeat in [1, 2]:
                    record = dict(valid=True, group_id='clip', task='duplex', provider=provider,
                                  round=repeat, response=response, scores={'final': 1.0})
                    (scores / f'{provider}_{repeat}.json').write_text(json.dumps(record))
            with patch.dict(os.environ, {}, clear=True), patch.object(evaluate, 'DATA_ROOT', data), \
                 patch.object(evaluate, 'objective_metrics', side_effect=AssertionError('Objective metrics must stay unused')), \
                 patch.object(score_mllm, 'run_many', side_effect=AssertionError('No API requests')), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main(['--metrics', 'mllm', '--mllm-score-dir', str(scores), '--output-dir', str(data / 'out')])
            summary = json.loads((data / 'out/summary.json').read_text())
            self.assertEqual(set(summary), {'mllm'})
            self.assertEqual(summary['mllm']['duplex']['mean'], 1.0)
            self.assertFalse((data / 'out/motion').exists())
            self.assertFalse((data / 'out/face').exists())

    def test_all_rejects_missing_online_settings_before_objective_work(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
             patch.object(evaluate, 'objective_metrics') as objective:
            with self.assertRaisesRegex(AssertionError, 'endpoint'):
                evaluate.main(['--metrics', 'all', '--prediction-dir', 'pred', '--smplx-model', 'skeleton',
                               '--flame-asset-dir', 'flame', '--output-dir', directory])
            objective.assert_not_called()


if __name__ == '__main__':
    unittest.main()
