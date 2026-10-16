"""Check raw-label selection and local source links used by released training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_paths import DATA_ROOT
sys.path.insert(0, str(DATA_ROOT))
from preprocess.link_training_data import FIELDS, PREFIX, link_training_data
from scripts import validate_release


class TrainingManifestTests(unittest.TestCase):
    def test_selected_manifest_has_one_raw_label_field(self):
        rows = json.loads((DATA_ROOT / 'manifests/train_selected81.json').read_text())['data']
        recipe = json.loads((ROOT / 'configs/training.json').read_text())
        self.assertEqual(len(rows), 1318)
        self.assertEqual(len({row['speaker_id'] for row in rows}), 81)
        self.assertEqual(recipe['data_args']['flame_file_key'], 'flame_file')
        for row in rows:
            self.assertEqual({key for key in row if key.startswith('flame_')}, {'flame_file'})
            self.assertTrue(row['flame_file'].endswith('_flame.npz'))

    def test_frozen_speaker_selection_replays_full_inventory(self):
        validate_release.validate_training_inventory()

    def test_source_links_are_repeatable_and_excluded_from_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data = root / 'upstream', root / 'data'
            row = {field: PREFIX + f'naturalistic/train/0000/{field}.npz' for field in FIELDS}
            for value in row.values():
                path = source / value[len(PREFIX):]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'original source bytes')
            manifest = data / 'manifests/train.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'data': [row]}))
            for _ in range(2):
                self.assertEqual(link_training_data(manifest, source, data), len(FIELDS))
            for value in row.values():
                link = data / value.removeprefix('data/')
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.read_bytes(), b'original source bytes')
            with patch.object(validate_release, 'DATA_ROOT', data):
                self.assertEqual(validate_release.payload_files(data), [manifest])

    def test_new_flame_labels_can_live_outside_the_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data, flame = root / 'upstream', root / 'data', root / 'new_flame'
            row = {field: PREFIX + f'improvised/train/0000/{field}.npz' for field in FIELDS}
            for field, value in row.items():
                path = (flame / Path(value).name if field == 'flame_file'
                        else source / value[len(PREFIX):])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'extracted label' if field == 'flame_file' else b'source')
            manifest = root / 'selected.json'
            manifest.write_text(json.dumps({'data': [row]}))
            self.assertEqual(link_training_data(manifest, source, data, flame), len(FIELDS))
            linked = data / row['flame_file'].removeprefix('data/')
            self.assertEqual(linked.resolve(), flame / Path(row['flame_file']).name)
            self.assertFalse((source / row['flame_file'][len(PREFIX):]).exists())

    def test_pairing_keeps_raw_labels_from_legacy_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, corrected = root / 'raw.npz', root / 'corrected.npz'
            raw.touch()
            corrected.touch()
            row = dict(pose_file='clip.npz', speaker_id='0001', flame_file=str(corrected),
                       flame_file_original=str(raw), flame_file_corrected=str(corrected),
                       flame_correction_accepted=True)
            source = root / 'source.json'
            source.write_text(json.dumps({'data': [row]}))
            output = root / 'paired.json'
            subprocess.run([sys.executable, str(DATA_ROOT / 'preprocess/attach_flame.py'),
                            '--input', str(source), '--flame-manifest', str(source), '--output', str(output)],
                           check=True, capture_output=True)
            paired = json.loads(output.read_text())['data'][0]
            self.assertEqual(paired['flame_file'], str(raw))
            self.assertEqual({key for key in paired if key.startswith('flame_')}, {'flame_file'})
            self.assertEqual(paired['speaker_id'], '0001')


if __name__ == '__main__':
    unittest.main()
