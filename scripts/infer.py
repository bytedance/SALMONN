#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def generate(model, processor, audio_paths, prompt, max_new_tokens=256):
    prompt_args = {"formatted_prompt": prompt} if "<audio>" in prompt else {"instruction": prompt}
    inputs = processor(audios=audio_paths, **prompt_args).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return processor.decode(output[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--audio", action="append", required=True, help="Repeat for multiple audio files")
    parser.add_argument("--prompt", default="Please describe the audio.")
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
    print(generate(model, processor, args.audio, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
