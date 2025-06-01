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
import random
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode. Visual dataset")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

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
            cur_len = cur_len if ('image' in sample) or ('video' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def process_image(self, image_file):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        if not isinstance(image_file, np.ndarray):
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
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
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image, image_size,"image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

        # still fail, most likely to be path issue or cloud disk issue, retry the same sample for longer
        for attempt_idx in range(num_final_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[final try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # Finally raise exception on failing.
        assert False, "Failed to fetch sample."

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        # print(self.list_data_dict[i]['id'])
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        suffix = None
        # import pdb;pdb.set_trace()
        if 'image' in sources[0]:
            # import pdb;pdb.set_trace()
            image_file = self.list_data_dict[i]['image']
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
            else:
                image = [self.process_image(image_file)]
            # import pdb;pdb.set_trace()
            num_frames = 0
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                num_frames
                )
            # breakpoint()
        elif 'video' in sources[0]:
            # import pdb;pdb.set_trace()
            video_file = self.list_data_dict[i]['video']
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
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
                sources, query_prompt = preprocess_multimodal_movie(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args, input_prompt)
            else:
                # import pdb;pdb.set_trace()
                try:
                    # using videoreader
                    if "shareVideoGPTV" not in video_file and "liangke" not in video_file:
                        # import pdb;pdb.set_trace()
                        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                        total_frame_num = len(vr)
                        avg_fps = round(vr.get_avg_fps()/self.data_args.video_fps)
                        frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
                        # sample_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
                        # frame_idx = [i for i in range(0, len(vr), sample_fps)]

                        # sample_fps = avg_fps
                        if self.data_args.frames_upbound > 0:
                            # import pdb;pdb.set_trace()
                            if len(frame_idx) > self.data_args.frames_upbound:
                                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.data_args.frames_upbound, dtype=int)
                                frame_idx = uniform_sampled_frames.tolist()
 
                        video = vr.get_batch(frame_idx).asnumpy()
                        # Convert the list of frames to a numpy array if needed
                        video = np.array(video)

                        num_frames_to_sample = num_frames = len(frame_idx)

                    else:
                        if "liangke" in video_file:
                            bytenas = os.getenv('BYTENAS')
                            if "vl-research-cn" not in bytenas:
                                video_file = self.list_data_dict[i]['video']
                            else:
                                video_file = os.path.join(video_folder, self.list_data_dict[i]['video'].split('/')[-1])
                        frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                        num_frames_to_sample = 10

                        total_frames = len(frame_files)

                        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

                        # Read and store the sampled frames
                        video = []
                        for idx in sampled_indices:
                            frame_path = frame_files[idx]
                            try:
                                with Image.open(frame_path) as img:
                                    # Convert the PIL image to a numpy array if needed
                                    # frame = np.array(img.convert('RGB'))
                                    frame = img.convert('RGB')
                                    video.append(frame)
                            except IOError:
                                print(f"Failed to read frame at path: {frame_path}")

                    processor = self.data_args.image_processor
                    image = processor.preprocess(video, return_tensors='pt')['pixel_values']
                    # import pdb;pdb.set_trace()
                    # image_tensors = []
                    # for f in video:
                    #     cur_image,cur_size = self.process_image(f)
                    #     image_tensors.append(cur_image)
                    # image_tensors = torch.stack(image_tensors)  
                    image = [(image, video[0].size,"video")]
                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args,
                        num_frames_to_sample)
                    # import pdb;pdb.set_trace()
                    # breakpoint()
                except:
                    import pdb;pdb.set_trace()
                    print(f"Failed to read video file: {video_file}")
                    return self._get_item(i+1)
                    # import pdb;pdb.set_trace()
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ('image' in self.list_data_dict[i]) or ('video' in self.list_data_dict[i])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt)

        if 'prompt' in data_dict:
            prompt = data_dict['prompt']
        else:
            prompt = None
        
        if suffix == 'pkl':
            prompt = [query_prompt]

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # import pdb;pdb.set_trace()
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif 'video' in self.list_data_dict[i]:
            # import pdb;pdb.set_trace()
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = [
                (
                    torch.zeros(1, 3, crop_size['height'], crop_size['width']),
                    (crop_size['width'], crop_size['height']),
                    "text"
                ),
            ]
        # prompt exist in the data
        # import pdb;pdb.set_trace()
        if prompt is not None:
            data_dict['prompt'] = prompt

        data_dict['id'] = self.list_data_dict[i]['id']

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels",'id'))
        # import pdb;pdb.set_trace()
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self.tokenizer.pad_token_id = 151643
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels,
                                   batch_first=True,
                                   padding_value=IGNORE_INDEX)
        # import pdb;pdb.set_trace()
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            ids=ids
        )

        if 'image' in instances[0]:
            # instances[1]['image'][0][0].shape
            # torch.Size([5, 3, 224, 224])
            images = [instance['image'] for instance in instances]
        
            batch['image_sizes'] = [im[1] for im_list in images for im in im_list]
            batch['modalities'] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]
            # import pdb;pdb.set_trace()
            

            if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # import pdb;pdb.set_trace()
        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

        return batch

