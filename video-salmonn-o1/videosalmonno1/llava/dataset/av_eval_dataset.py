from llava.dataset.preprocess_utils import preprocess_multimodal, preprocess, preprocess_multimodal_movie
import json
from torch.utils.data import Dataset
import math
import pickle
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from dataclasses import dataclass, field
from PIL import Image
import time
import copy
import os
from typing import Dict, Optional, Sequence, List
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
import numpy as np
import transformers
import torch 
from transformers import WhisperFeatureExtractor
import random
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence


class LazyAVEvalDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 test_split=None):
        super(LazyAVEvalDataset, self).__init__()
        if data_args.val_path is None:
            if test_split is not None:
                list_data_dict = test_split
                print('Testing Data Size:', len(list_data_dict))
            else:
                list_data_dict = json.load(open(data_path, "r"))
                random.seed(2024)
                random.shuffle(list_data_dict)
                self.test_data = list_data_dict[0 : int(data_args.val_ratio * len(list_data_dict))]
                list_data_dict = list_data_dict[int(data_args.val_ratio * len(list_data_dict)) + 1:]
                print('Training Data Size:', len(list_data_dict))
        else:
            list_data_dict = json.load(open(data_path, "r"))

        self.random_video = data_args.random_video
        if test_split is not None:
            self.random_video = False
        if self.random_video:
            self.all_video = [item['video'] for item in list_data_dict if 'video' in item]
            random.seed(2024)
    
        print("Formatting inputs...Skip in lazy mode. Audio visual dataset")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        whisper_path = self.data_args.audio_processor
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.max_time = data_args.max_time

        non_key_frames = self.data_args.extra_frames
        self.max_frame_num = round(self.max_time * self.data_args.video_fps * non_key_frames)
        print("Max frame num: ", self.max_frame_num)

    def get_test_set(self):
        return self.test_data

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # cur_len = cur_len if 'image' in sample else -cur_len
            cur_len = cur_len if ('image' in sample) or ('video' in sample) or ("audio" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def process_image(self, image_file):
        
        image_mode = "image" # modality determine
        
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        if not isinstance(image_file, np.ndarray):
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = Image.open(image_file).convert('RGB')
        else:
            image = Image.fromarray(image_file).convert('RGB')
        image_size = image.size
        if self.data_args.image_aspect_ratio == 'highres':
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif self.data_args.image_aspect_ratio == 'anyres':
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif self.data_args.image_aspect_ratio == 'crop_split':
            image = process_highres_image_crop_split(image, self.data_args)
        elif self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        elif self.data_args.image_aspect_ratio == 'fake_video':
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # copy n images to form a fake video
            random_frames = random.randint(5, 16)
            image = torch.stack([image]*random_frames, dim=0)
            image_mode = "video"
            # breakpoint()
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image, image_size, image_mode # image_mode is 'image' for all `image_aspect_ratio` except fake_video

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else

        num_base_retries = 300

        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[try #{attempt_idx}] Failed to fetch sample {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
                time.sleep(1)

        assert False, "Failed to fetch sample."
        # num_base_retries = 3
        # num_final_retries = 300

        # # try the current sample first
        # for attempt_idx in range(num_base_retries):
        #     try:
        #         sample = self._get_item(i)
        #         return sample
        #     except Exception as e:
        #         # sleep 1s in case it is a cloud disk issue
        #         print(f'[try #{attempt_idx}] Failed to fetch sample {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
        #         time.sleep(1)

        # # try other samples, in case it is file corruption issue
        # for attempt_idx in range(num_base_retries):
        #     try:
        #         sample_idx = random.choice(range(len(self)))
        #         sample = self._get_item(sample_idx)
        #         return sample
        #     except Exception as e:
        #         # no need to sleep
        #         print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
        #         pass

        # # still fail, most likely to be path issue or cloud disk issue, retry the same sample for longer
        # for attempt_idx in range(num_final_retries):
        #     try:
        #         sample = self._get_item(i)
        #         return sample
        #     except Exception as e:
        #         # sleep 1s in case it is a cloud disk issue
        #         print(f'[final try #{attempt_idx}] Failed to fetch sample {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
        #         time.sleep(1)

        # # Finally raise exception on failing.
        # assert False, "Failed to fetch sample."

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if self.random_video:
            if 'image' not in sources[0] and 'video' not in sources[0] and 'audio' in sources[0]:
                sources[0] = copy.deepcopy(self.list_data_dict[i])
                sources[0]['video'] = random.choice(self.all_video)


        suffix = None
        # breakpoint()
        if 'image' in sources[0]:
            # breakpoint()
            image_file = sources[0]['image']
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
            else:
                image = [self.process_image(image_file)]
            num_frames = 0
            process_sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                num_frames
            )
        elif 'video' in sources[0]:
            # breakpoint()
            video_file = sources[0]['video']
            # video_folder = self.data_args.video_folder
            # video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split('.')[-1]
            if not os.path.exists(video_file):
                print('File {} not exist!'.format(video_file))
            
            if suffix == 'pkl':
                video_info = pickle.load(open(video_file, 'rb'))
                image = torch.from_numpy(video_info['feats'][:, 1:])
                input_prompt = video_info['inputs'].replace('...', '')
                # replace the default image token with multiple tokens
                input_prompt = input_prompt.replace(DEFAULT_IMAGE_TOKEN, 
                                                    DEFAULT_IMAGE_TOKEN * self.data_args.video_token)
                process_sources, query_prompt = preprocess_multimodal_movie(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args, input_prompt)
            else:
                # breakpoint()
                try:
                    if False: # ('sharevideogptv' in video_file.lower()) and ('frame' in video_file.lower()):
                        frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                        num_frames_to_sample = 10

                        total_frames = len(frame_files)

                        if total_frames>num_frames_to_sample:
                            sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
                        else:
                            sampled_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)

                        # Read and store the sampled frames
                        video = []
                        for idx in sampled_indices:
                            frame_path = frame_files[idx]
                            try:
                                with Image.open(frame_path) as img:
                                    frame = img.convert('RGB')
                                    video.append(frame)
                            except IOError:
                                print(f"Failed to read frame at path: {frame_path}")
                    else:
                        # breakpoint()
                        if False: # video_file.endswith('.webm') or 'ego4d' in video_file:
                            # Use VideoFileClip from moviepy: handle '.webm' videos
                            non_key_frames = self.data_args.extra_frames

                            clip = VideoFileClip(video_file)
                            total_frames = clip.reader.nframes
                            fps = round(clip.reader.fps / non_key_frames)
                            duration = clip.duration

                            if total_frames / fps > self.max_frame_num:
                                fps = round(total_frames / self.max_frame_num)

                            frames = []
                            for i, frame in enumerate(clip.iter_frames()):
                                if i % fps == 0 and len(frames) < math.ceil(duration): # get the 1st frame of each second
                                    frames.append(frame)

                            if len(frames) % non_key_frames != 0:
                                available_len = len(frames) - len(frames) % non_key_frames
                                frames = frames[:available_len]

                            video = np.stack(frames) # video: (F, H, W, C)
                            num_frames = video.shape[0]
                            video_mask = np.ones(video.shape[0], dtype=bool)
                            video_mask[::non_key_frames] = False
                            non_key_video = video
                            video = video[~video_mask]

                            # real_fps = fps
                            real_time = duration
                        else:
                            non_key_frames = self.data_args.extra_frames

                            # Use VideoReader from decord
                            # breakpoint()
                            try:
                                try:
                                    vr = VideoReader(video_file, ctx=cpu(0))
                                    if self.data_args.full_framerate:
                                        non_key_frames = round(vr.get_avg_fps() / 3)
                                    total_frame_num = len(vr)
                                    avg_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
                                    real_time = total_frame_num / vr.get_avg_fps()
                                    
                                    max_frames = self.max_frame_num
                                    frame_idx = [k for k in range(0, total_frame_num, round(avg_fps / non_key_frames))]
                                    if len(frame_idx) > max_frames:
                                        frame_idx = [k for k in range(0, total_frame_num, round(total_frame_num / max_frames))]

                                    if len(frame_idx) % non_key_frames != 0:
                                        available_len = len(frame_idx) - len(frame_idx) % non_key_frames
                                        frame_idx = frame_idx[:available_len]

                                    video = vr.get_batch(frame_idx).asnumpy() # video: (F, H, W, C)
                                    video = np.array(video)

                                except:
                                    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                                    if self.data_args.full_framerate:
                                        non_key_frames = round(vr.get_avg_fps() / 3)
                                    total_frame_num = len(vr)
                                    avg_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
                                    real_time = total_frame_num / vr.get_avg_fps()
                                    
                                    max_frames = self.max_frame_num
                                    frame_idx = [k for k in range(0, total_frame_num, round(avg_fps / non_key_frames))]
                                    if len(frame_idx) > max_frames:
                                        frame_idx = [k for k in range(0, total_frame_num, round(total_frame_num / max_frames))]

                                    if len(frame_idx) % non_key_frames != 0:
                                        available_len = len(frame_idx) - len(frame_idx) % non_key_frames
                                        frame_idx = frame_idx[:available_len]

                                    video = vr.get_batch(frame_idx).asnumpy() # video: (F, H, W, C)
                                    video = np.array(video)

                            except:
                                clip = VideoFileClip(video_file)

                                if self.data_args.full_framerate:
                                    non_key_frames = round(clip.reader.fps / 3)
                                total_frames = clip.reader.nframes
                                fps = round(clip.reader.fps / non_key_frames)
                                duration = clip.duration

                                if total_frames / fps > self.max_frame_num:
                                    fps = round(total_frames / self.max_frame_num)

                                frames = []
                                signal.signal(signal.SIGALRM, handler)
                                signal.alarm(10)
                                try:
                                    for k, frame in enumerate(clip.iter_frames()):
                                        if k % fps == 0: # get the 1st frame of each second
                                            frames.append(frame)
                                except Exception as e:
                                    print(e)
                                
                                signal.alarm(0)

                                if len(frames) % non_key_frames != 0:
                                    available_len = len(frames) - len(frames) % non_key_frames
                                    frames = frames[:available_len]

                                video = np.stack(frames) # video: (F, H, W, C)
                                real_time = duration

                            video_mask = np.ones(video.shape[0], dtype=bool)
                            video_mask[::non_key_frames] = False
                            non_key_video = video
                            video = video[~video_mask]

                    processor = self.data_args.image_processor
                    image = processor.preprocess(video, return_tensors='pt')['pixel_values']
                    frames = processor.preprocess(non_key_video, return_tensors='pt')['pixel_values']
                    image = [(image, video[0].size, "video")]
                    frames = [(frames, non_key_video[0].size, "video")]
                    process_sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args,
                        num_frames=image[0][0].shape[0]
                    )
                    # breakpoint()
                except Exception as e:
                    print(f"Failed to read video file: {video_file}. Line: {e.__traceback__.tb_lineno}, Exception:", e)
                    return self._get_item(i + 1)
                    # breakpoint()
        else:
            # sources = copy.deepcopy([e["conversations"] for e in sources])
            process_sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                num_frames=0, # image[0][0].shape[0]
            )

        if 'audio' in sources[0]:
            audio_file = sources[0]['audio']
            audio, sr = sf.read(audio_file)
            if len(audio.shape) == 2: # stereo to mono
                audio = audio[:, 0]

            org_audio = audio
            if True:
                org_audio = org_audio[:, np.newaxis]

            if len(audio) < sr: # pad audio to at least 1s
                sil = np.zeros(sr - len(audio), dtype=float)
                audio = np.concatenate((audio, sil), axis=0)

            if 'video' in sources[0]:
                audio = audio[:round(sr * real_time)] 
            else:
                audio = audio[:round(sr * self.max_time)] # truncate audio to at most 30s
            audio_lst = [audio[k: k + 30 * sr] for k in range(0, len(audio), 30 * sr)]
            spectrogram_lst = [self.wav_processor(a, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze() for a in audio_lst]
            org_audio_lst = [org_audio[k: k + 30 * sr, :] for k in range(0, len(audio), 30 * sr)]

        else:
            audio_file = None
            sr = 16000
            org_audio_lst = None
            if 'video' in sources[0]:
                audio = np.zeros(round(sr * real_time))
            else:
                audio = np.zeros(sr * 10)

            audio_lst = [audio[k: k + 30 * sr] for k in range(0, len(audio), 30 * sr)]
            spectrogram_lst = [self.wav_processor(a, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze() for a in audio_lst]


        has_image = ('image' in sources[0]) or ('video' in sources[0]) or ('audio' in sources[0])
        if "video" in sources[0]:
            data_id = "['{}', '{}']".format(sources[0]["video"], audio_file)
        else:
            data_id = "['{}', '{}']".format(None, audio_file)
            
        data_dict = preprocess(
            process_sources,
            self.tokenizer,
            has_image=has_image,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt
        )

        if 'prompt' in data_dict:
            prompt = data_dict['prompt']
        else:
            prompt = None
        
        if suffix == 'pkl':
            prompt = [query_prompt]

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], reject_input_ids=data_dict["reject_input_ids"][0], reject_labels=data_dict["reject_labels"][0])

        # breakpoint()
        # image exist in the data
        if 'image' in sources[0]:
            data_dict['image'] = image
            data_dict['frames'] = frames
            data_dict['modality'] = "image"
        elif 'video' in sources[0]:
            # breakpoint()
            data_dict['image'] = image
            data_dict['frames'] = frames
            if 'audio' not in sources[0]:
                data_dict['modality'] = "video"
            else:
                data_dict['modality'] = "audio-video"
        # elif self.data_args.is_multimodal:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = self.data_args.image_processor.crop_size
        #     data_dict['image'] = [
        #         (
        #             torch.zeros(1, 3, crop_size['height'], crop_size['width']),
        #             (crop_size['width'], crop_size['height']),
        #             "text"
        #         ),
        #     ]
        else:
            data_dict['modality'] = "audio"
            data_dict['image'] = None

        data_dict["raw_wav"] = audio_lst
        data_dict["spectrogram"] = torch.stack(spectrogram_lst, dim=0)
        data_dict["multi_channel_wav"] = org_audio_lst

        # prompt exist in the data
        # breakpoint()
        if prompt is not None:
            data_dict['prompt'] = prompt

        if data_dict['modality'] != "audio":
            data_dict["real_time"] = real_time
        else:
            data_dict["real_time"] = 30 * len(audio_lst)

        data_dict['id'] = data_id # "['{}', '{}']".format(sources[0]['video'], audio_file)
        
        return data_dict

