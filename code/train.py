"""Train the gesture or gesture-expression stage from the released recipe."""
import argparse
import json
from pathlib import Path
from train_ellsa2 import main
ROOT = Path(__file__).resolve().parent

def main_release():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["gesture", "gesture_expression"], required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/training.json"))
    args, overrides = parser.parse_known_args()
    from release_paths import resolve_config
    config = resolve_config(json.loads(Path(args.config).read_text()), args.stage)
    flat = {k: v for section in ("model_args", "data_args", "training_args") for k, v in config[section].items()}
    # Let the original parser apply CLI overrides after JSON defaults.
    import sys
    cli = []
    for key, value in flat.items():
        if value is None:
            continue
        if isinstance(value, dict):
            cli += ["--" + key, json.dumps(value)]
        elif isinstance(value, list):
            if key == "report_to" and not value:
                cli += ["--report_to", "none"]
            elif value:
                cli += ["--" + key] + [str(x) for x in value]
        else:
            cli += ["--" + key, str(value).lower() if isinstance(value, bool) else str(value)]
    sys.argv = [sys.argv[0]] + cli + overrides
    import os
    os.chdir(ROOT)
    main()

if __name__ == "__main__":
    main_release()
