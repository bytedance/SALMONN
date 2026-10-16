"""Verify asset upload boundaries and Git rules without loading model weights."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from scripts import validate_release

KAOLRM_FILES = ('config.json', 'model.safetensors', 'LICENSE_WEIGHT.txt')
LOCAL_FILES = (
    'checkpoints/external_assets/smplh/SMPLH_FEMALE.pkl',
    'checkpoints/external_assets/smplx/SMPLX_NEUTRAL_2020.npz',
    'checkpoints/external_assets/flame/flame_2020.pt',
    'checkpoints/external_assets/kaolrm-mono/unknown.txt',
    'checkpoints/external_assets/kaolrm-mono/cache/config.json',
    'checkpoints/external_assets/notes.txt',
    'code/external_assets/model.safetensors',
    'data/external_assets/model.safetensors',
    'data/source/local.wav',
    'checkpoints/outputs/output.npz',
)


def asset_fixture(root):
    """Create bundled files alongside geometry, local assets, and runtime outputs."""
    bundled = [f'checkpoints/external_assets/kaolrm-mono/{name}' for name in KAOLRM_FILES]
    for relative in [*bundled, *LOCAL_FILES]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    return set(bundled)


class ReleaseAssetTests(unittest.TestCase):
    def test_manifest_includes_only_bundled_assets(self):
        """Both aggregate and checkpoint manifests must exclude restricted geometry."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = asset_fixture(root)
            with patch.object(validate_release, 'ROOT', root), \
                 patch.object(validate_release, 'DATA_ROOT', root / 'data'), \
                 patch.object(validate_release, 'CHECKPOINT_ROOT', root / 'checkpoints'):
                aggregate = validate_release.payload_files()
                checkpoints = validate_release.payload_files(root / 'checkpoints')
                self.assertEqual({str(path.relative_to(root)) for path in aggregate}, expected)
                self.assertEqual(aggregate, checkpoints)
                self.assertEqual(validate_release.payload_files(root / 'code'), [])
                self.assertEqual(validate_release.payload_files(root / 'data'), [])

    def test_git_upload_rules_match_manifest_boundary(self):
        """Package Git ignores must allow KaoLRM, exclude geometry, and use LFS for weights."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundled = asset_fixture(root)
            # Data/checkpoints are independent repositories; the aggregate root excludes them.
            for package in ('code', 'data', 'checkpoints'):
                for name in ('.gitignore', '.gitattributes'):
                    source = CODE.parent / package / name
                    if source.exists():
                        shutil.copyfile(source, root / package / name)
                subprocess.run(['git', 'init', '--quiet', str(root / package)], check=True, capture_output=True)
            def ignored(relative):
                package, name = relative.split('/', 1)
                return subprocess.run(['git', 'check-ignore', '--no-index', name],
                                      cwd=root / package, capture_output=True).returncode
            for relative in sorted(bundled):
                self.assertEqual(ignored(relative), 1, relative)
            for relative in LOCAL_FILES:
                self.assertEqual(ignored(relative), 0, relative)
            weights = 'external_assets/kaolrm-mono/model.safetensors'
            result = subprocess.check_output(['git', 'check-attr', 'filter', '--', weights],
                                             cwd=root / 'checkpoints', text=True)
            self.assertEqual(result.strip(), weights + ': filter: lfs')
            config = 'external_assets/kaolrm-mono/config.json'
            result = subprocess.check_output(['git', 'check-attr', 'filter', '--', config],
                                             cwd=root / 'checkpoints', text=True)
            self.assertEqual(result.strip(), config + ': filter: unspecified')

    def test_bundled_assets_reject_symlinks(self):
        """A bundled filename must not silently upload a linked private asset."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_fixture(root)
            weights = root / 'checkpoints/external_assets/kaolrm-mono/model.safetensors'
            weights.unlink()
            weights.symlink_to(root / 'checkpoints/external_assets/notes.txt')
            with patch.object(validate_release, 'CHECKPOINT_ROOT', root / 'checkpoints'):
                with self.assertRaisesRegex(AssertionError, 'Symlink in release payload'):
                    validate_release.payload_files(root / 'checkpoints')


if __name__ == '__main__':
    unittest.main()
