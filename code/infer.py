"""Published gesture + expression inference entry point."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent

RELEASE_ARGS = [
    "--dialog", "true", "--speaker-id-type", "motion", "--motion-tokenizer-type", "upper_hands",
    "--num-motion-codebooks", "18", "--gesture-num-codebooks", "12", "--expression-num-codebooks", "6",
    "--motion-style-expression", "true",
]

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--test-id")
    p.add_argument("--test-limit", type=int, default=0)
    p.add_argument("--output-dir")
    p.add_argument("--model-path", "--model_path", help="Generator checkpoint directory, including tokenizer files")
    args, extra = p.parse_known_args()
    command = [sys.executable, str(ROOT / "scripts/infer_ellsa2_gesture_expression_from_audio.py"), *RELEASE_ARGS]
    if args.model_path:
        command += ["--model-path", str(Path(args.model_path).expanduser().resolve())]
    if args.output_dir:
        extra += ["--output-dir", str(Path(args.output_dir).resolve())]
    if args.test_id or args.test_limit:
        from scripts.prepare_testset import prepare
        dest = Path(args.output_dir).resolve() if args.output_dir else ROOT / "outputs/test"
        job_file = prepare(dest, args.test_id, args.test_limit)
        command += ["--audio-json", str(job_file), "--output-dir", str(dest)]
    subprocess.run(command + extra, check=True, cwd=ROOT)

if __name__ == "__main__":
    main()
