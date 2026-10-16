"""Check release assets, private metadata and the complete distributable file list."""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import struct
import zipfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_paths import CODE_ROOT, DATA_ROOT, CHECKPOINT_ROOT
ROOT = CODE_ROOT.parent
RUNTIME_DIRS = {"outputs", "runs", "cache", "__pycache__", ".git", "external_assets"}
# Match private infrastructure and credential formats without printing their values.
PRIVATE = re.compile(
    rb"/mnt/(?:bn|bd)/|/home/[A-Za-z][^/\s\x00]{1,40}/|/roo[t]/|"
    rb"byted(?:ance)?[.](?:net|org|com)|jump[-]proxy|fd[b]d:|"
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:sk-|hf_)[A-Za-z0-9_-]{24,}"
)
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".txt", ".sh", ".yaml", ".yml", ".csv", ".html", ".log", ".jinja", ".js"}


def payload_files(root=None):
    """List upload files, including KaoLRM but excluding licensed geometry and runtime products."""
    root = ROOT if root is None else root
    files = []
    assets = CHECKPOINT_ROOT / "external_assets"
    for directory, children, names in os.walk(root):
        directory = Path(directory)
        children[:] = sorted(name for name in children if name not in RUNTIME_DIRS
                             or (name == "external_assets" and directory == CHECKPOINT_ROOT))
        # Geometry licenses restrict redistribution; whitelist only the KaoLRM release files.
        if directory == assets:
            children[:] = [name for name in children if name == "kaolrm-mono"]
            names = []
        elif directory == assets / "kaolrm-mono":
            children[:] = []
            names = [name for name in names if name in {
                "config.json", "model.safetensors", "LICENSE_WEIGHT.txt"}]
        if directory == DATA_ROOT:
            children[:] = [name for name in children if name != "source"]
        for name in children:
            path = Path(directory) / name
            assert not path.is_symlink(), f"Symlink in release payload: {path.relative_to(root)}"
        for name in names:
            if name in RUNTIME_DIRS:
                continue
            path = Path(directory) / name
            assert not path.is_symlink(), f"Symlink in release payload: {path.relative_to(root)}"
            if name != "MANIFEST.sha256" and path.suffix != ".pyc":
                files.append(path)
    return sorted(files)


def check_private_metadata(path):
    def check(data, member=""):
        assert not PRIVATE.search(data), f"Private metadata in {path.name} {member}"
        # Unicode NPY strings use UTF-32; removing NULs exposes the same text.
        if member.endswith(".npy"):
            assert not PRIVATE.search(data.replace(b"\x00", b"")), f"Private array metadata in {path.name} {member}"

    if path.suffix in TEXT_SUFFIXES or path.name.startswith("."):
        check(path.read_bytes())
    elif path.suffix in {".pth", ".pt", ".npz"}:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                # Inspect serialized metadata, never deserialize checkpoint pickles.
                if entry.filename.endswith(".pkl"):
                    check(archive.read(entry), entry.filename)
                elif entry.filename.endswith(".npy"):
                    with archive.open(entry) as handle:
                        header = handle.read(512)
                    if re.search(rb"['\"]descr['\"]:\s*['\"][<>=|]?[OUS]", header):
                        check(archive.read(entry), entry.filename)
    elif path.suffix == ".safetensors":
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            check(handle.read(size))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_training_inventory():
    sys.path.insert(0, str(DATA_ROOT))
    from preprocess.build_stageA_select_spk import select_rows, build_train_samples
    from preprocess.common import infer_speaker_id
    pool = json.loads((DATA_ROOT / "manifests/train_all_resolved.json").read_text())["data"]
    rows = json.loads((DATA_ROOT / "manifests/train_selected81.json").read_text())["data"]
    expected = json.loads((DATA_ROOT / "manifests/selected_speakers.json").read_text())
    with (DATA_ROOT / "manifests/speaker_quality.csv").open() as handle:
        selected = select_rows(list(csv.DictReader(handle)))
    assert len(pool) == 121054
    assert len({infer_speaker_id(row) for row in pool}) == 3401
    assert len(rows) == 1318 and len(expected) == len(set(expected)) == 81
    assert {row["speaker_id"] for row in selected} == set(expected)
    assert {row["speaker_id"] for row in rows} == set(expected)
    replay = build_train_samples(pool, selected)
    assert replay == [{k: v for k, v in row.items() if k != "flame_file"} for row in rows]


def validate_data():
    validate_training_inventory()
    rows = [json.loads(s) for s in (DATA_ROOT / "testset/manifest.jsonl").read_text().splitlines() if s]
    assert len(rows) == len({r["group_id"] for r in rows}) == 138
    assert collections.Counter(r["human_approved_task"] for r in rows) == {"duplex": 67, "semantic": 71}
    protocol = json.loads((DATA_ROOT / "testset/protocol.json").read_text())
    assert len({row["session_id"] for row in rows}) == protocol["sessions"]
    assert collections.Counter(row["source_kind"] for row in rows) == protocol["source_counts"]
    assert collections.Counter(row["human_approved_scene"] for row in rows
                               if row["human_approved_task"] == "duplex") == protocol["human_duplex_categories"]
    assert all([row["rating_start_sec"], row["rating_end_sec"]] == [4, 16] for row in rows)
    context = json.loads((DATA_ROOT / "testset/judge_context.json").read_text())
    assert len(context) == len(rows)
    assert {entry["group_id"]: entry["task"] for entry in context} == {
        row["group_id"]: row["human_approved_task"] for row in rows}
    assert all((DATA_ROOT / entry["metadata"]["reference_video"]).is_file() for entry in context)
    for row in rows:
        for value in row.values():
            if isinstance(value, str) and value.startswith("testset/"):
                assert (DATA_ROOT / value).is_file(), value


def validate_checkpoints():
    """Check required generator, RVQ tokenizer, and bundled KaoLRM files."""
    for name in ("config.json", "model.safetensors", "LICENSE_WEIGHT.txt"):
        assert (CHECKPOINT_ROOT / "external_assets/kaolrm-mono" / name).is_file(), name
    generators = ["standalone_generator"]
    if (CHECKPOINT_ROOT / "standalone_generator_guided").exists():
        generators.append("standalone_generator_guided")
    for name in generators:
        with (CHECKPOINT_ROOT / name / "model.safetensors").open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        assert {int(k.split(".")[1]) for k in header if k.startswith("motion_heads.")} == set(range(18))
        assert any(k.startswith("audio_encoder.") for k in header)
    base = CHECKPOINT_ROOT / "base_asr_model"
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        assert (base / name).is_file(), name
    with (base / "model.safetensors").open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    assert {int(k.split(".")[1]) for k in header if k.startswith("motion_heads.")} == set(range(6))
    assert any(k.startswith("audio_encoder.") for k in header)
    for name in ("upper", "hands", "flame"):
        assert (CHECKPOINT_ROOT / "tokenizers" / (name + ".pth")).is_file()


def verify_manifest(root, files, hashes):
    expected = {}
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        value, name = line.split("  ", 1)
        assert name not in expected, f"Duplicate manifest entry: {name}"
        expected[name] = value
    actual = {str(path.relative_to(root)): hashes[path] for path in files}
    assert set(expected) == set(actual), f"Manifest coverage mismatch: {set(expected) ^ set(actual)}"
    for name, value in actual.items():
        assert expected[name] == value, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=["all", "code", "data", "checkpoints"], default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checksums", action="store_true", help="Verify exact payload coverage and SHA256 in each package")
    mode.add_argument("--write-manifest", action="store_true", help="Regenerate package and complete-release manifests")
    args = parser.parse_args()
    if args.package in {"all", "data"}:
        validate_data()
    if args.package in {"all", "checkpoints"}:
        validate_checkpoints()
    roots = {"code": CODE_ROOT, "data": DATA_ROOT, "checkpoints": CHECKPOINT_ROOT}
    selected = roots if args.package == "all" else {args.package: roots[args.package]}
    groups = [(name, root, payload_files(root)) for name, root in selected.items()]
    if args.package == "all" and DATA_ROOT == ROOT / "data" and CHECKPOINT_ROOT == ROOT / "checkpoints":
        groups.append(("release", ROOT, payload_files()))
    hashes = {}
    for name, root, files in groups:
        for path in files:
            if path not in hashes:
                if path.suffix == ".py":
                    compile(path.read_text(), str(path), "exec")
                check_private_metadata(path)
                hashes[path] = digest(path) if args.checksums or args.write_manifest else None
        if args.checksums:
            verify_manifest(root, files, hashes)
        elif args.write_manifest:
            (root / "MANIFEST.sha256").write_text("".join(
                f"{hashes[path]}  {path.relative_to(root)}\n" for path in files))
        print(f"Validated {name}: {len(files)} payload files; no private metadata or external symlinks")


if __name__ == "__main__":
    main()
