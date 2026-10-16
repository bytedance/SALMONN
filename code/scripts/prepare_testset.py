"""Code-side entry point for the data package's test-job preparation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_paths import DATA_ROOT
sys.path.insert(0, str(DATA_ROOT))
from preprocess.prepare_testset import prepare, read_manifest, main

if __name__ == "__main__":
    main()
