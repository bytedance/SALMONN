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
from libero_utils import save_rollout_gif, get_libero_image, get_episode_length, get_libero_wrist_image, quat2axisangle
from libero_utils import get_libero_dummy_action, get_libero_env
from libero.libero import benchmark

import json
import torchaudio
import kaldifeat
from lhotse import Fbank, FbankConfig
import math
import random
import time
random.seed(int(time.time()))

ELLSA_BASE_PATH = os.environ.get("ELLSA_BASE_PATH")
ELLSA_DATA_PATH = os.environ.get("ELLSA_DATA_PATH")
COSY_CKPT_PATH = os.environ.get("COSY_CKPT_PATH")
LLAMA_CKPT_PATH = os.environ.get("LLAMA_CKPT_PATH")
VISION_VQ_PATH = os.environ.get("VISION_VQ_PATH")

sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference/cosyvoice/third_party/Matcha-TTS"))
sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference/Emu3"))
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

def setup():
    dist.init_process_group(backend="nccl")
    os.environ["EGL_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def evaluate(
    model,
    model_name=None,
    ckpt_name=None,
    debug=False,
    resize_size=256,
    num_trials_per_task=50, #50,
    num_steps_wait=10,
    local_log_dir=None,
    task_suite_name="libero_object",
    speech_task_suite_name="llama_questions",
    speech_start=0,
    speech_end=None,
    speech=False,
    stop=False,
    context_vqa=False,
    silence=False,
    limit=False,
    refuse=False,
    encoder_type="mamba",
    predict_action_frames=10,
    generate=False
):
    # Initialize Local Logging
    run_id = f"{task_suite_name}-{time.strftime('%Y-%m-%d_%H:%M')}"
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logger.info(f"CKPT: {ckpt_name}")
    log_file.write(f"CKPT: {ckpt_name}\n")
    logger.info(f"Task suite: {task_suite_name}")
    log_file.write(f"Task suite: {task_suite_name}\n")
    EP_LEN = get_episode_length(task_suite_name)
    if speech:
        if refuse:
            with open(os.path.join(ELLSA_DATA_PATH,"json/refuse_command.json"), "r") as f:
                instruction_data = json.load(f)
        else:
            with open(os.path.join(ELLSA_DATA_PATH,"json/libero_eval_speech.json"), 'r') as f:
                instruction_data = json.load(f)
        
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
    
    if limit:
        robo_tasks = [9] # [2,3,5,9]
        questions = [14,63,46,70,7]
    
    if refuse:
        refuse_types = ["visual","semantic","motion","out-of-context"]
        split = int(num_trials_per_task / len(refuse_types))
    
    if stop:
        stop_base_dir = os.path.join(ELLSA_DATA_PATH,"interrupt/test")
        stop_speech = os.listdir(stop_base_dir)
    elif context_vqa:
        with open(os.path.join(ELLSA_DATA_PATH,"json/eval_context_vqa.json"), 'r') as f:
            context_vqa_data = json.load(f)
    elif silence or refuse:
        pass
    else:
        speech_idx = 0
        if speech_task_suite_name == "llama_questions":
            with open(os.path.join(ELLSA_DATA_PATH,"json/llama_questions.json"), "r") as f:
                speech_data = json.load(f)["annotation"][speech_start:speech_end]
        elif speech_task_suite_name == "web_questions":
            with open(os.path.join(ELLSA_DATA_PATH,"json/web_questions.json"), "r", encoding="utf-8") as f:
                speech_data = json.load(f)["annotation"][speech_start:speech_end]
        elif speech_task_suite_name == "triviaQA":
            with open(os.path.join(ELLSA_DATA_PATH,"json/triviaqa.json"), "r", encoding="utf-8") as f:
                speech_data = json.load(f)["annotation"][speech_start:speech_end]
        elif speech_task_suite_name == "alpaca_eval":
            with open(os.path.join(ELLSA_DATA_PATH,"json/alpacaeval.json"), "r") as f:
                speech_data = json.load(f)["annotation"][speech_start:speech_end]
        num_trials_per_task = max(num_trials_per_task, math.ceil(len(speech_data) / num_tasks_in_suite))

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in range(num_tasks_in_suite):
        if limit:
            if task_id not in robo_tasks:
                continue
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)
        task_episodes, task_successes = 0, 0
        logger.info(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")
        if speech:
            if refuse:
                refuse_type_num = 0
                audio4stream, fs = torchaudio.load(instruction_data[task_suite_name][str(task_id)][refuse_types[refuse_type_num]][0]["path"])
            else:
                audio4stream, fs = torchaudio.load(instruction_data[task_suite_name][str(task_id)])
            speech_len = math.ceil(audio4stream.shape[1] / fs * 10)
        if context_vqa:
            repeat_num = len(context_vqa_data[task_suite_name][str(task_id)])
        else: 
            repeat_num = 1

        for episode_idx in range(repeat_num * num_trials_per_task):
            env.reset()
            model.reset()

            if refuse and episode_idx and episode_idx % split == 0:
                refuse_type_num += 1
                audio4stream, fs = torchaudio.load(instruction_data[task_suite_name][str(task_id)][refuse_types[refuse_type_num]][0]["path"])
                speech_len = math.ceil(audio4stream.shape[1] / fs * 10)

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            if model.use_cot:
                thought = [""]

            # Start episodes
            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")
            if speech:
                if stop:
                    sample = os.path.join(stop_base_dir,random.choice(stop_speech))
                    speech_question, fs = torchaudio.load(sample)
                    if fs != 16000:
                        resample_transform = torchaudio.transforms.Resample(fs, 16000)
                        fs = 16000
                        speech_question = resample_transform(speech_question)
                    speech_len_speech = math.ceil(speech_question.shape[1] / fs * 10)
                    test_len = 1
                elif context_vqa:
                    # sample = random.choice(context_vqa_data[task_suite_name][str(task_id)])
                    sample = context_vqa_data[task_suite_name][str(task_id)][episode_idx%repeat_num]
                    speech_question, fs = torchaudio.load(sample)
                    if fs != 16000:
                        resample_transform = torchaudio.transforms.Resample(fs, 16000)
                        fs = 16000
                        speech_question = resample_transform(speech_question)
                    speech_len_speech = math.ceil(speech_question.shape[1] / fs * 10)
                    test_len = 1
                elif silence or refuse:
                    speech_question = torch.zeros((1, fs))
                    speech_len_speech = 0
                    test_len = 1
                else:
                    if limit:
                        item = speech_data[questions[int(np.floor(episode_idx/len(questions)))]]
                    else:
                        # item = random.choice(speech_data)
                        item = speech_data[speech_idx % len(speech_data)]
                        speech_idx += 1
                    speech_question, fs = torchaudio.load(item["path"][0])
                    speech_len_speech = math.ceil(speech_question.shape[1] / fs * 10)
                    test_len = 1.6

                if context_vqa:
                    middle_break = random.randint(10,220)
                else:
                    if limit:
                        middle_break = 20 + (episode_idx%5) * 10
                    else:
                        middle_break = random.randint(predict_action_frames*2,predict_action_frames*4)
                pad_audio = torch.zeros(
                    (1, int(middle_break/predict_action_frames * fs))
                )
                pad_audio2 = torch.zeros(
                    (1, int(test_len*EP_LEN/predict_action_frames * fs))
                )
                audio4stream_used = torch.cat(
                    (audio4stream, pad_audio, speech_question, pad_audio2), dim=1
                )
                if stop:
                    print(f"time: {speech_len+middle_break+speech_len_speech}, stop sample: {sample}\n")
                    log_file.write(f"time: {speech_len+middle_break+speech_len_speech}, stop sample: {sample}\n")
                elif context_vqa:
                    print(f"time: {speech_len+middle_break+speech_len_speech}, vqa sample: {sample}\n")
                    log_file.write(f"time: {speech_len+middle_break+speech_len_speech}, vqa sample: {sample}\n")
                elif not silence and not refuse:
                    print(f"interval: {middle_break}, Q:{item['Q']}...\n")
                    log_file.write(f"interval: {middle_break}, Q:{item['Q']}...\n")

                audio4stream_used = [audio4stream_used.squeeze()]
                if encoder_type == "mamba":
                    fbank_feature = fbank(audio4stream_used)[0]
                elif encoder_type == "zipformer2":
                    fbank_feature = fbank.extract(audio4stream_used[0], sampling_rate=fs)
                fbank_feature_len = fbank_feature.size(0)
                turn_taking = True
            hyp = ""
            action_counter = 0
            current_done = False
            while t < test_len * EP_LEN + num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # Prepare observation
                    observation, img = prepare_observation(obs, resize_size)
                    replay_images.append(img)
                    
                    if model.use_cot:
                        # Create a white background for the text
                        text_img = (
                            np.ones((img.shape[0], 1000, 3), dtype=np.uint8) * 255
                        )
                        # Split thought into multiple lines
                        lines = thought[0].replace("@", "\n").split("\n")
                        # Add text lines
                        for i, line in enumerate(lines):
                            cv2.putText(
                                text_img,
                                line,
                                (10, 30 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                1,
                            )
                        # Concatenate original image with text image
                        img = np.concatenate((img, text_img), axis=1)
                        # Save a sample image for debugging
                        # cv2.imwrite("sample_cot_image.png", img)

                    if action_counter == 0:
                        if model.use_cot:
                            action, thought = model.step(obs_img, task_description)
                        else:
                            if speech:
                                text, action = model.step_withspeech(observation, fbank_feature, fbank_feature_len, t - num_steps_wait, generate_text=True)
                                hyp += text
                                if text == "<silence><eot>" and t > speech_len+middle_break+speech_len_speech+50+num_steps_wait and current_done:
                                    break
                                if text == "<silence><eot>" and t > speech_len * 2 + num_steps_wait and refuse:
                                    break
                            else:
                                action = model.step(observation, task_description)
                            # from PIL import Image
                            # Image.fromarray(img).save(f"img_{t}_{action_counter}.png")
                            # Image.fromarray(observation['wrist_image']).save(f"wrist_{t}_{action_counter}.png")

                        action_counter = action.shape[0]

                    
                    # logger.info(f"Action: {action.shape}")
                    step_action = action[-action_counter]
                    # if speech:
                    #    if t < speech_len + num_steps_wait and step_action.tolist() != [0, 0, 0, 0, 0, 0, -1]:
                    #        turn_taking = False
                    obs, reward, done, info = env.step(step_action.tolist())
                    action_counter -= 1
                    if done and not current_done:
                        print(f"action end: {t}\n")
                        log_file.write(f"action end: {t}\n")
                        task_successes += 1
                        total_successes += 1
                        current_done = True
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1
            if current_done == False:
                print(f"action end: {t}\n")
                log_file.write(f"action end: {t}\n")

            # Save a replay video of the episode
            logger.info(f"Num of Steps: {len(replay_images)}")
            if debug and len(replay_images) > 0:
                gif_dir = os.path.join(local_log_dir, "videos-{}".format(run_id))
                if not os.path.exists(gif_dir):
                    os.makedirs(gif_dir, exist_ok=True)
                gif_path = f"Episodes{total_episodes}_{str(current_done)}.gif"
                gif_path = os.path.join(gif_dir, gif_path)
                save_rollout_gif(replay_images, gif_path, fps=15)

            # Log current results
            logger.info(f"Success: {current_done}")
            # if speech:
            #    logger.info(f"Turn-taking: {turn_taking}")
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )
            log_file.write(f"Success: {current_done}\n")
            # if speech:
            #    log_file.write(f"Turn-taking: {turn_taking}")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            if not silence and not stop and not context_vqa and not refuse:
                ref_text = item["text"]
                logger.info(f"\nReference: {ref_text}")
                log_file.write(f"\nReference: {ref_text}\n")
            logger.info(
                f"Hypothesis: {hyp}\n"
                )
            log_file.write(
                f"Hypothesis: {hyp}\n"
            )
            log_file.flush()

            if generate:
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
                        if stop:
                            wav_name = f"task{str(task_id)}_episode{str(episode_idx)}"
                        elif context_vqa:
                            wav_name = f"task{str(task_id)}_episode{str(episode_idx)}"
                        else:
                            wav_name = item["path"][0].split("/")[-1].split(".")[0]
                        sf.write(os.path.join(wav_path, f"{wav_name}.wav"), ret_wav.squeeze().detach().cpu().numpy(), 24000)

        # Log final results
        logger.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logger.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()

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
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        help="select evaluate LIBREO TASK SUITE",
    )
    parser.add_argument(
        "--speech_task_suite_name",
        type=str,
        choices=[
            "llama_questions",
            "web_questions",
            "web_questions_part1",
            "web_questions_part2",
            "web_questions_left",
            "triviaQA",
            "alpaca_eval"
        ],
        help="select evaluate speech QA TASK SUITE",
    )
    parser.add_argument("--speech", default=False, type=bool, help="whether to use speech")
    parser.add_argument("--moe", default=False, type=bool, help="whether to use moe")
    parser.add_argument("--generate", default=False, type=bool, help="whether to generate speech")
    parser.add_argument("--speech_start", default=0, type=int, help="speech data start idx")
    parser.add_argument("--speech_end", default=None, type=int, help="speech data end idx")
    parser.add_argument("--attn_adapter", default=False, type=bool, help="whether to use attn adapter")
    parser.add_argument("--attn_adapter_type", default="None", type=str, help="the type of attn adapter")
    parser.add_argument("--time_block", default=1.0, type=float, help="time block of full-duplex modeling")
    parser.add_argument("--merge_speech_lora", default=False, type=bool, help="whether to merge lora of speech expert")
    parser.add_argument("--lora_modules", default="default", type=str, help="the target modulrs of lora")
    parser.add_argument("--predict_action_frames", default=10, type=int, help="predict number of action frames pre step")
    parser.add_argument("--stop", default=False, type=bool, help="whether to test stop")
    parser.add_argument("--silence", default=False, type=bool, help="whether to test silence")
    parser.add_argument("--limit", default=False, type=bool, help="whether to limited test")
    parser.add_argument("--refuse", default=False, type=bool, help="whether to test refuse")
    parser.add_argument("--context_vqa", default=False, type=bool, help="whether to test context vqa")
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
    if not args.no_nccl:
        setup()

    CACHE_ROOT = args.cache_root
    os.makedirs(CACHE_ROOT, exist_ok=True)

    eval_log_dir = os.path.join(CACHE_ROOT, 'release')
    os.makedirs(eval_log_dir, exist_ok=True)

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    model = EmuVLAModel(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=torch.device("cuda"),
            speech=args.speech,
            moe=args.moe,
            generate=args.generate,
            attn_adapter=args.attn_adapter,
            attn_adapter_type=args.attn_adapter_type,
            merge_speech_lora=args.merge_speech_lora,
            lora_modules=args.lora_modules,
            encoder_type=args.encoder_type,
            time_block=args.time_block,
            predict_action_frames=args.predict_action_frames
        )

    sr_path = os.path.join(eval_log_dir, f"success_rate_calvin.txt")
    result_path = os.path.join(
        eval_log_dir, f"results_calvin_rand-{args.rank}.json"
    )
    cache_file = os.path.join(eval_log_dir, f"meta_info.json")

    if not args.no_cache and args.local_rank == 0:
        if os.path.exists(cache_file):
            os.remove(cache_file)
        with open(cache_file, "w") as f:
            _info = {
                "eval_sr_path": sr_path,
                "eval_result_path": result_path,
                "eval_log_dir": eval_log_dir,
            }
            json.dump(_info, f, indent=2)

    evaluate(
        model,
        task_suite_name=args.task_suite_name,
        speech_task_suite_name=args.speech_task_suite_name,
        speech_start=args.speech_start,
        speech_end=args.speech_end,
        ckpt_name=args.emu_hub,
        local_log_dir=eval_log_dir,
        debug=args.debug,
        speech=args.speech,
        stop=args.stop,
        context_vqa=args.context_vqa,
        silence=args.silence,
        limit=args.limit,
        refuse=args.refuse,
        encoder_type=args.encoder_type,
        predict_action_frames=args.predict_action_frames,
        generate=args.generate
    )

    if not args.no_nccl:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    main()
