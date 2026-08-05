#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from infer import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor.prepare_model(model)
    samples = json.loads(Path(args.input).read_text())
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for sample in samples:
            response = generate(model, processor, sample["audios"], sample["prompt"], args.max_new_tokens)
            handle.write(json.dumps({**sample, "response": response}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
