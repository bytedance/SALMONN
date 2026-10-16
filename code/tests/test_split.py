"""Verify independently downloaded packages and data-root-relative job paths."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_paths import DATA_ROOT


class SplitTests(unittest.TestCase):
    def test_download_roots_are_independent_of_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(os.environ, ELLSA2_DATA_ROOT=str(root / 'dataset-snapshot'),
                       ELLSA2_CHECKPOINT_ROOT=str(root / 'model-snapshot'))
            program = (
                'import sys,json;sys.path.insert(0,sys.argv[1]);'
                'from release_paths import resolve_config;'
                'print(json.dumps(resolve_config(json.load(open(sys.argv[2])), "expression")))')
            config = json.loads(subprocess.check_output([
                sys.executable, '-I', '-c', program, str(ROOT),
                str(ROOT / 'configs/training.json')], cwd=root, env=env, text=True))
            self.assertEqual(config['model_args']['model_name_or_path'],
                             str(root / 'model-snapshot/base_asr_model'))
            self.assertEqual(config['data_args']['data_path'],
                             str(root / 'dataset-snapshot/manifests/train_selected81.json'))
            self.assertEqual(config['data_args']['flame_stats_path'],
                             str(ROOT / 'assets/flame_stats.json'))
            self.assertEqual(config['training_args']['deepspeed'],
                             str(ROOT / 'configs/deepspeed_zero2.json'))

    def test_data_preparation_needs_no_model_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / 'dataset-snapshot'
            (downloaded / 'testset').mkdir(parents=True)
            row = json.loads((DATA_ROOT / 'testset/manifest.jsonl').read_text().splitlines()[0])
            (downloaded / 'testset/manifest.jsonl').write_text(json.dumps(row) + '\n')
            program = (
                'import runpy,sys;'
                'sys.argv=[sys.argv[1],"--test-id","fro_00038","--output-dir",sys.argv[2]];'
                'runpy.run_path(sys.argv[0],run_name="__main__");'
                'assert "torch" not in sys.modules and "dataset" not in sys.modules')
            subprocess.run([sys.executable, '-I', '-c', program,
                            str(DATA_ROOT / 'preprocess/prepare_testset.py'), str(root / 'jobs')],
                           cwd=root, env=dict(os.environ, ELLSA2_DATA_ROOT=str(downloaded)),
                           check=True, capture_output=True)
            jobs = json.loads((root / 'jobs/inference_jobs.json').read_text())['data']
            self.assertEqual(jobs[0]['audio_file'], str(downloaded / row['audio20_target']))
            self.assertEqual(jobs[0]['motion_style_pose_path'],
                             str(downloaded / row['motion_style_pose_path']))


if __name__ == '__main__':
    unittest.main()
