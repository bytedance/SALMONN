#!/bin/bash

# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/baaivision/UniVLA. The original license is located at 'third-party-license/UniVLA.txt'.

import traceback
import argparse
import json
import logging
from pathlib import Path
import time
from utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
import sys
import os
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from pytorch_lightning import seed_everything
import torch
import torch.distributed as dist
from utils.config_utils import load_config

from model_wrapper_emu import EmuVLAModel

import json
import torchaudio
import kaldifeat
from lhotse import Fbank, FbankConfig
import math

ELLSA_BASE_PATH = os.environ.get("ELLSA_BASE_PATH")
ELLSA_DATA_PATH = os.environ.get("ELLSA_DATA_PATH")
COSY_CKPT_PATH = os.environ.get("COSY_CKPT_PATH")
LLAMA_CKPT_PATH = os.environ.get("LLAMA_CKPT_PATH")
VISION_VQ_PATH = os.environ.get("VISION_VQ_PATH")

sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference/cosyvoice/third_party/Matcha-TTS"))
sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference"))
from cosyvoice.cli.cosyvoice import CosyVoice2
import soundfile as sf

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s - %(name)s - %(levelname)s - %(message)s]"
)
logger = logging.getLogger(__name__)

def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def evaluate(
    model,
    model_name=None,
    ckpt_name=None,
    debug=False,
    resize_size=256,
    num_trials_per_task=10, #50,
    num_steps_wait=10,
    local_log_dir=None,
    task_suite_name="librispeech_testclean",
    speech=False,
    generate=False,
    encoder_type="mamba",
    time_block=1.0
):
    # Initialize Local Logging
    run_id = f"{task_suite_name}-{time.strftime('%Y-%m-%d_%H:%M')}"
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    if task_suite_name == "llama_questions":
        with open(os.path.join(ELLSA_DATA_PATH,"json/llama_questions.json"), "r") as f:
            speech_data = json.load(f)["annotation"]
        repeat_num = 12
    elif task_suite_name == "web_questions":
        with open(os.path.join(ELLSA_DATA_PATH,"json/web_questions.json"), "r", encoding="utf-8") as f:
            speech_data = json.load(f)["annotation"]
        repeat_num = 12
    elif task_suite_name == "triviaQA":
        with open(os.path.join(ELLSA_DATA_PATH,"json/triviaqa.json"), "r", encoding="utf-8") as f:
            speech_data = json.load(f)["annotation"]
        repeat_num = 12
    elif task_suite_name == "alpaca_eval":
        with open(os.path.join(ELLSA_DATA_PATH,"json/alpacaeval.json"), "r") as f:
            speech_data = json.load(f)["annotation"]
        repeat_num = 20
    logger.info(f"CKPT: {ckpt_name}")
    log_file.write(f"CKPT: {ckpt_name}\n")
    logger.info(f"Task suite: {task_suite_name}")
    log_file.write(f"Task suite: {task_suite_name}\n")
    if speech:    
        if encoder_type == "mamba":
            opts = kaldifeat.FbankOptions()
            opts.frame_opts.dither = 0
            opts.frame_opts.snip_edges = False
            opts.frame_opts.samp_freq = 16000 # only support 16k audio
            opts.mel_opts.num_bins = 80 # 80-bin
            opts.mel_opts.high_freq = -400

            fbank = kaldifeat.Fbank(opts)
        elif encoder_type == "zipformer2":
            fbank = Fbank(FbankConfig(num_mel_bins=128))
    
    if generate:
        cosy_frontend = CosyVoice2(COSY_CKPT_PATH, load_jit=False, load_trt=False).frontend
        wav_path = os.path.join(local_log_dir, run_id)
        os.makedirs(wav_path, exist_ok=True)

        prompt_audio, fs = torchaudio.load(os.path.join(ELLSA_BASE_PATH,"reference/RoboVLMs/wav/emo_10004_b_0.wav"))
        prompt_speech_feat, prompt_speech_feat_len = cosy_frontend._extract_speech_feat(prompt_audio.to(model.device))
        prompt_audio = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(prompt_audio.cpu()).to(model.device)
        spk_embedding = cosy_frontend._extract_spk_embedding(prompt_audio)
        prompt_speech_token, prompt_speech_token_len = cosy_frontend._extract_speech_token(prompt_audio)

    # Start evaluation
    for item in speech_data:
        # Initialize LIBERO environment and task description
        model.reset()
        ref_text = item["text"]
        logger.info(f"\nReference: {ref_text}")
        log_file.write(f"\nReference: {ref_text}\n")
        if speech:
            audio4stream, fs = torchaudio.load(item["path"][0])
            frames = int(time_block * fs)
            speech_len = math.ceil(audio4stream.shape[1] / frames)
            max_inference_nums = min(int(speech_len * repeat_num),int(300 / time_block)) # max 385
            pad_audio = torch.zeros(
                (1, int(max_inference_nums * frames))
            )
            audio4stream = torch.cat(
                (audio4stream, pad_audio), dim=1
            )
            audio4stream = [audio4stream.squeeze()]
            if encoder_type == "mamba":
                fbank_feature = fbank(audio4stream)[0]
            elif encoder_type == "zipformer2":
                fbank_feature = fbank.extract(audio4stream[0], sampling_rate=fs)
            fbank_feature_len = fbank_feature.size(0)
            turn_taking = True

        t = 0
        hyp = ""
        while t < max_inference_nums:
            with torch.no_grad():
                step_generated = model.step_onlyspeech(fbank_feature, fbank_feature_len, t, item["task"])
            hyp += step_generated
            if step_generated == "<silence><eot>" and t > speech_len * 2:
                break
            t += 1

        # Log final results
        logger.info(
            f"Hypothesis: {hyp}\n"
        )
        log_file.write(
            f"Hypothesis: {hyp}\n"
        )
        log_file.flush()

        if generate and "librispeech" not in task_suite_name:
            if len(model.tts_features) > 0:
                gen_wavs = []

                for feature in model.tts_features:
                    final = False
                    embedded_y = None
                    text_idx_num = 0
                    num_generated = 0
                    end_idx = model.model.speech_expert.generator.llm.text_chunk if model.moe else model.model.generator.llm.text_chunk
                    while not final:
                        text_token = feature[:, :end_idx].to(model.device)
                        text_token_lens = torch.tensor([text_token.shape[1]]).to(text_token.device)

                        with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
                            try:
                                if model.moe:
                                    if embedded_y is None:
                                        embedded_y, generated, cur_generated, final = model.model.speech_expert.generator.llm.inference_salmonn_stream(text_token, text_token_lens, text_idx_num=text_idx_num, token_input=embedded_y)
                                    else:
                                        embedded_y, generated, cur_generated, final = model.model.speech_expert.generator.llm.inference_salmonn_stream(text_token, text_token_lens, text_idx_num=text_idx_num, token_input=embedded_y, generated=generated)
                                else:
                                    if embedded_y is None:
                                        embedded_y, generated, cur_generated, final = model.model.generator.llm.inference_salmonn_stream(text_token, text_token_lens, text_idx_num=text_idx_num, token_input=embedded_y)
                                    else:
                                        embedded_y, generated, cur_generated, final = model.model.generator.llm.inference_salmonn_stream(text_token, text_token_lens, text_idx_num=text_idx_num, token_input=embedded_y, generated=generated)
                            except:
                                break

                        text_idx_num += model.model.speech_expert.generator.llm.text_chunk if model.moe else model.model.generator.llm.text_chunk
                        end_idx += model.model.speech_expert.generator.llm.text_chunk if model.moe else model.model.generator.llm.text_chunk

                        if end_idx > 100:
                            final = True

                        if final:
                            with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
                                hift_cache = None
                                # step = 50
                                step = 5000
                                if model.moe:
                                    while(num_generated + step + 3 < len(generated)):
                                        if num_generated == 0:
                                            new_gen_wav, hift_cache, tts_mel = model.model.speech_expert.generator.tts_salmonn(generated[num_generated:num_generated+step], spk_embedding, flow_prompt_speech_token=prompt_speech_token, prompt_speech_feat=prompt_speech_feat, finalize=False, hift_cache = None)
                                            num_generated += step - 3
                                        else:
                                            new_gen_wav, hift_cache, tts_mel = model.model.speech_expert.generator.tts_salmonn(generated[num_generated:num_generated+step+3], spk_embedding, flow_prompt_speech_token=prompt_speech_token, prompt_speech_feat=prompt_speech_feat, finalize=False, hift_cache = hift_cache)
                                            num_generated += step
                                        gen_wavs.append(new_gen_wav.clone())
                                    new_gen_wav, _, _ = model.model.speech_expert.generator.tts_salmonn(generated[num_generated:], spk_embedding, flow_prompt_speech_token=prompt_speech_token, prompt_speech_feat=prompt_speech_feat, hift_cache=hift_cache, finalize=True)
                                else:
                                    while(num_generated + step + 3 < len(generated)):
                                        if num_generated == 0:
                                            new_gen_wav, hift_cache, tts_mel = model.model.generator.tts_salmonn(generated[num_generated:num_generated+step], spk_embedding, flow_prompt_speech_token=prompt_speech_token, prompt_speech_feat=prompt_speech_feat, finalize=False, hift_cache = None)
                                            num_generated += step - 3
                                        else:
                                            new_gen_wav, hift_cache, tts_mel = model.model.generator.tts_salmonn(generated[num_generated:num_generated+step+3], spk_embedding, flow_prompt_speech_token=prompt_speech_token, prompt_speech_feat=prompt_speech_feat, finalize=False, hift_cache = hift_cache)
                                            num_generated += step
                                        gen_wavs.append(new_gen_wav.clone())
                                    new_gen_wav, _, _ = model.model.generator.tts_salmonn(generated[num_generated:], spk_embedding, flow_prompt_speech_token=prompt_speech_token, prompt_speech_feat=prompt_speech_feat, hift_cache=hift_cache, finalize=True)
                                gen_wavs.append(new_gen_wav.clone())

                if len(gen_wavs) > 0:
                    ret_wav = torch.cat(gen_wavs, dim=1)
                    wav_name = item["path"][0].split("/")[-1].split(".")[0]
                    sf.write(os.path.join(wav_path, f"{wav_name}.wav"), ret_wav.squeeze().detach().cpu().numpy(), 24000)

    log_file.close()

def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Prepare observations dict
    observation = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img 

def parser_args():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info and visualize environment.",
    )

    # yaml_path takes the highest priority, then the log_dir, finally the config_path
    parser.add_argument(
        "--config_path", type=str, default=None, help="path to the config file"
    )
    parser.add_argument(
        "--is_pt_config",
        action="store_true",
        help="whether the specified config path is a pretrain config file.",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        nargs="+",
        default="",
        help="checkpoint directory of the training",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="checkpoint directory of the training",
    )
    parser.add_argument(
        "--ckpt_idx", type=int, default=-1, help="which ckpt is going to be evaluated"
    )
    parser.add_argument("--emu_hub", type=str, default="")
    parser.add_argument("--vq_hub", type=str, default="")
    parser.add_argument("--vision_hub", type=str, default=VISION_VQ_PATH)
    parser.add_argument("--encoder_type", type=str, default="mamba")
    parser.add_argument(
        "--task_suite_name",
        type=str,
        help="select evaluate LIBREO TASK SUITE",
    )
    parser.add_argument("--speech", default=False, type=bool, help="whether to use speech")
    parser.add_argument("--moe", default=False, type=bool, help="whether to use moe")
    parser.add_argument("--mix", default=False, type=bool, help="whether to use mix")
    parser.add_argument("--generate", default=False, type=bool, help="whether to generate speech")
    parser.add_argument("--attn_adapter", default=False, type=bool, help="whether to use attn adapter")
    parser.add_argument("--attn_adapter_type", default="None", type=str, help="the type of attn adapter")
    parser.add_argument("--merge_speech_lora", default=False, type=bool, help="whether to merge lora of speech expert")
    parser.add_argument("--lora_modules", default="default", type=str, help="the target modulrs of lora")
    parser.add_argument("--time_block", default=1.0, type=float, help="time block of full-duplex modeling")
    parser.add_argument("--predict_action_frames", default=10, type=int, help="predict number of action frames pre step")
    parser.add_argument("--device_id", default=0, type=int, help="CUDA device")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--debug_model", action="store_true")
    parser.add_argument("--no_nccl", action="store_true")
    parser.add_argument("--no_action_ensemble", action="store_true")
    parser.add_argument('--cache_root', type=str, default="",
                        help="Root directory to store cache/logs.")

    args = parser.parse_args()
    return args


def main():
    args = parser_args()

    CACHE_ROOT = args.cache_root
    os.makedirs(CACHE_ROOT, exist_ok=True)

    eval_log_dir = os.path.join(CACHE_ROOT, 'speech')
    os.makedirs(eval_log_dir, exist_ok=True)

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    model = EmuVLAModel(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=torch.device("cuda"),
            speech=args.speech,
            moe=args.moe,
            mix=args.mix,
            attn_adapter=args.attn_adapter,
            attn_adapter_type=args.attn_adapter_type,
            merge_speech_lora=args.merge_speech_lora,
            lora_modules=args.lora_modules,
            generate=args.generate,
            encoder_type=args.encoder_type,
            time_block=args.time_block,
            predict_action_frames=args.predict_action_frames
        )

    evaluate(
        model,
        task_suite_name=args.task_suite_name,
        ckpt_name=args.emu_hub,
        local_log_dir=eval_log_dir,
        debug=args.debug,
        speech=args.speech,
        generate=args.generate,
        encoder_type=args.encoder_type,
        time_block=args.time_block
    )

    if not args.no_nccl:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    main()