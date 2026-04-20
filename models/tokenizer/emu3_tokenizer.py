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

import os
import os.path as osp
import torch
from transformers import AutoModel, AutoImageProcessor
import numpy as np
import sys
import json
import re
import time
from tqdm import tqdm
import argparse
from PIL import Image, ImageEnhance
import random

def random_shift(images, max_shift=0.1):
    """
    对一个图像列表进行随机平移
    :param images: List of PIL Images (视频的每一帧)
    :param max_shift: 最大平移比例
    :return: 平移后的图像列表
    """
    width, height = images[0].size
    shift_x = random.uniform(-max_shift, max_shift) * width
    shift_y = random.uniform(-max_shift, max_shift) * height
    
    # 对每一帧图像应用相同的平移
    shifted_images = [
        image.transform(
            (width, height),
            Image.AFFINE,
            (1, 0, shift_x, 0, 1, shift_y),
            resample=Image.BICUBIC
        ) for image in images
    ]
    
    return shifted_images

def random_color_enhance(images):
    """
    对一个图像列表进行随机颜色增强
    :param images: List of PIL Images (视频的每一帧)
    :return: 颜色增强后的图像列表
    """
    factor = random.uniform(0.5, 1.5)  # 随机调整颜色的增强因子
    
    color_enhanced_images = [
        ImageEnhance.Color(image).enhance(factor) for image in images
    ]
    
    return color_enhanced_images

def random_brightness_enhance(images):
    """
    对一个图像列表进行随机亮度增强
    :param images: List of PIL Images (视频的每一帧)
    :return: 亮度增强后的图像列表
    """
    factor = random.uniform(0.5, 1.5)  # 随机调整亮度的增强因子
    
    brightness_enhanced_images = [
        ImageEnhance.Brightness(image).enhance(factor) for image in images
    ]
    
    return brightness_enhanced_images

def load_images(folder_path, size, interval=2, augmentation=None):
    """Load and resize all images in the specified folder."""
    image_paths = sorted(os.listdir(folder_path),key=natural_sort_key)[::interval]
    images = [Image.open(osp.join(folder_path, img)).resize(size) for img in image_paths]

    if augmentation is not None:
        images = augmentation(images)

    return images,image_paths

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def load_images_from_json(json_path, video_root, size):
    """Load and resize images based on JSON paths."""
    with open(json_path, 'r') as f:
        image_paths = json.load(f)['images'][::2]
    images = [Image.open(osp.join(video_root, v)).resize(size) for v in image_paths]
    image_names = [osp.basename(v).split('.')[0] for v in image_paths]
    return images, image_names

def clip_level_enc_dec(images, model, processor, save_codes_path, save_recon_path, t=4):
    """Process images in batches and save codes and reconstructed images."""
    images_tensor = processor(images, return_tensors="pt")["pixel_values"].unsqueeze(0).cuda()
    num_frames = images_tensor.shape[1]
    num_batches = num_frames // t

    os.makedirs(save_codes_path, exist_ok=True)
    os.makedirs(save_recon_path, exist_ok=True)

    for batch in range(num_batches):
        image = images_tensor[:, batch * t : (batch + 1) * t]
        with torch.no_grad():
            codes = model.encode(image)
            x = codes.detach().cpu().numpy()
            np.save(f'{save_codes_path}/{batch:03d}.npy', x)
            recon = model.decode(codes)
        recon = recon.view(-1, *recon.shape[2:])
        recon_images = processor.postprocess(recon)["pixel_values"]
        for idx, im in enumerate(recon_images):
            im.save(f"{save_recon_path}/{batch * t + idx:03d}.jpg")

def image_level_enc_dec(images, model, processor, save_codes_path, save_recon_path, batch_size=1):
    """Process images in batches: encode, decode, and save the reconstructed images and codes."""
    os.makedirs(save_codes_path, exist_ok=True)  # Ensure the codes directory exists
    # os.makedirs(save_recon_path, exist_ok=True)
    images_tensor = processor(images, return_tensors="pt")["pixel_values"].cuda()
    num_images = images_tensor.shape[0]
    for start_idx in range(0, num_images, batch_size):
        batch = images_tensor[start_idx:start_idx + batch_size]
        try:
            with torch.no_grad():
                # Encode the batch of images
                codes = model.encode(batch)
                # Decode the codes back to images
                recon = model.decode(codes)
                # Save the encoded codes
                np.save(f'{save_codes_path}/{start_idx:03d}.npy', codes.detach().cpu().numpy())
            
            # recon = recon.view(-1, *recon.shape[2:])
            # recon_images = processor.postprocess(recon)["pixel_values"]
            # for idx, im in enumerate(recon_images):
            #     im.save(f"{save_recon_path}/image_{start_idx + idx:03d}.jpg")
        except Exception as e:
            print(f"Error processing batch starting at image {start_idx}: {e}")


def image_level_enc(images, image_paths, model, processor, save_codes_path, batch_size=8):
    """Process images in batches: encode and save the codes."""
    t1 = time.time()
    os.makedirs(save_codes_path, exist_ok=True)  # Ensure the codes directory exists
    images_tensor = processor(images, return_tensors="pt")["pixel_values"].cuda()
    num_images = images_tensor.shape[0]
    t2 = time.time()
    print(f"Time to load images: {(t2 - t1) * 1000} ms")
    for start_idx in range(0, num_images, batch_size):
        batch = images_tensor[start_idx:start_idx + batch_size]
        try:
            with torch.no_grad():
                t1 = time.time()
                # Encode the batch of images
                codes = model.encode(batch)
                t2 = time.time()
                print(f"Time to encode: {(t2 - t1) * 1000} ms")
                # Save the encoded codes
                for idx, code in enumerate(codes):
                    np.save(f'{save_codes_path}/{image_paths[start_idx + idx]}.npy', code.detach().cpu().numpy())
                t3 = time.time()
                print(f"Time to save codes: {(t3 - t2) * 1000} ms")
        except Exception as e:
            print(f"Error processing batch starting at image {start_idx}: {e}")

data_config = {
    'DROID': {
        'view': 'wrist_image_left',
        'min_pixels': 128 * 256,
        'interval': 1,
        'SIZE': (256, 144),
        'hz_func': lambda interval: 15 // interval,
        'VIDEO_ROOT': lambda view: f'/share/project/yuqi.wang/datasets/processed_data/droid_fast/{view}',
        'VIDEO_CODES_SAVE': lambda view, hz: f'/share/project/yuqi.wang/datasets/post_train_data/droid/{view}_codes_256_{hz}hz',
        'VIDEO_RECON_SAVE': lambda view, hz: f'/share/project/yuqi.wang/datasets/post_train_data/droid/{view}_recon_256_{hz}hz'
    },
    'RT1': {
        'min_pixels': 128 * 128,
        'interval': 1,
        # 'SIZE': (160, 128),
        'SIZE': (240, 192),
        'hz_func': lambda interval: 3 // interval,
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/rt1',
        'VIDEO_CODES_SAVE': lambda size, hz: f'/share/project/yuqi.wang/datasets/post_train_data/rt1_codes_{size[0]}_{hz}hz',
        'VIDEO_RECON_SAVE': lambda size, hz: f'/share/project/yuqi.wang/datasets/post_train_data/rt1_recon_{size[0]}_{hz}hz'
    },
    '1X': {
        'min_pixels': 128 * 128,
        'interval': 15,
        'SIZE': (192, 192),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/1x-wm/video_format',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/processed_data/1x_codes_192_2hz',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/processed_data/1x_recon_192_2hz'
    },
    'BridgeV2': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (240, 180),
        'hz_func': lambda interval: 5 // interval,
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/bridgev2',
        'VIDEO_CODES_SAVE': lambda size, hz: f'/share/project/yuqi.wang/datasets/post_train_data/bridgev2_codes_{size[0]}_{hz}hz',
        'VIDEO_RECON_SAVE': lambda size, hz: f'/share/project/yuqi.wang/datasets/post_train_data/bridgev2_codes_{size[0]}_{hz}hz'
    },
    'Calvin': {
        'min_pixels': 80 * 80,
        'interval': 1,
        'SIZE': (80, 80),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/calvin',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/processed_data/calvin_gripper_codes_augshift',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/processed_data/calvin_recon'
    },
    'Calvin_partial': {
        'min_pixels': 80 * 80,
        'interval': 1,
        # 'SIZE': (80, 80),
        'SIZE': (200,200),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/calvin_partial',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/processed_data/calvin_partial_codes',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/processed_data/calvin_partial_recon'
    },
    'Calvin_raw': {
        'min_pixels': 80 * 80,
        'interval': 1,
        'SIZE': (80, 80),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/calvin_raw_video/rgb_gripper',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/processed_data/calvin_raw_gripper_codes',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/processed_data/calvin_raw_recon'
    },
    'Ego5m': {
        'min_pixels': 128 * 128,
        'interval': 5,
        'SIZE': (256, 144),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/ego5m',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/processed_data/ego5m_256_codes',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/processed_data/ego5m_256_recon'
    },
    'libero': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (200, 200),
        'VIDEO_ROOT': '/mnt/bn/audio-visual-llm-data5/datasets/modified_libero_rlds/libero_all',
        'VIDEO_CODES_SAVE': '/mnt/bn/audio-visual-llm-data5/datasets/modified_libero_rlds/libero_all_codes_200_augshift',
        'VIDEO_RECON_SAVE': '/mnt/bn/audio-visual-llm-data5/datasets/modified_libero_rlds/libero_recon_256'
    },
    'bridge_orig': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/bridge',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/sft_data/bridge_orig_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/sft_data/bridge_orig_recon'
    },
    'maniskill': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/maniskill',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/maniskill_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/maniskill_recon'
    },
    'fmb': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/fmb',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/fmb_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/fmb_recon'
    },
    'toto': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/toto',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/toto_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/toto_recon'
    },
    'taco_play': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/taco_play',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/taco_play_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/taco_play_recon'
    },
    'kuka': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/kuka',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/kuka_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/kuka_recon'
    },
    'berkeley_autolab_ur5': {
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/berkeley_autolab_ur5',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/berkeley_autolab_ur5_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/berkeley_autolab_ur5_recon_256'
    },
    'viola':{
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/viola',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/viola_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/viola_recon_256'
    },
    'cmu_play_fusion':{
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/cmu_play_fusion',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/cmu_play_fusion_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/cmu_play_fusion_recon_256'
    },
    'utaustin_mutex':{
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/oxembodiment/utaustin_mutex',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/utaustin_mutex_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/utaustin_mutex_recon_256'
    },    
    'SSv2':{
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (256, 256),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/processed_data/SSv2',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/SSv2_codes_256',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/post_train_data/SSv2_recon_256'
    },
    'aloha_songling':{
        'min_pixels': 128 * 128,
        'interval': 1,
        'SIZE': (128, 128),
        'VIDEO_ROOT': '/share/project/yuqi.wang/datasets/real_robot/songling_official_data/锅盖放在架子上',
        'VIDEO_CODES_SAVE': '/share/project/yuqi.wang/datasets/sft_data/aloha_songling/pot_lid_on_shelf_codes_128',
        'VIDEO_RECON_SAVE': '/share/project/yuqi.wang/datasets/sft_data/aloha_songling_recon_128'
    },
}

def get_data_config(process_data):
    cfg = data_config[process_data]

    interval = cfg['interval']
    size = cfg['SIZE']
    min_pixels = cfg['min_pixels']
    hz = cfg['hz_func'](interval) if 'hz_func' in cfg else None

    if process_data == 'DROID':
        view = cfg['view']
        video_root = cfg['VIDEO_ROOT'](view)
        video_codes_save = cfg['VIDEO_CODES_SAVE'](view, hz)
        video_recon_save = cfg['VIDEO_RECON_SAVE'](view, hz)
    elif process_data == 'RT1':
        video_root = cfg['VIDEO_ROOT']
        video_codes_save = cfg['VIDEO_CODES_SAVE'](size, hz)
        video_recon_save = cfg['VIDEO_RECON_SAVE'](size, hz)
    elif process_data == 'BridgeV2':
        video_root = cfg['VIDEO_ROOT']
        video_codes_save = cfg['VIDEO_CODES_SAVE'](size, hz)
        video_recon_save = cfg['VIDEO_RECON_SAVE'](size, hz)
    elif process_data == 'aloha':
        task = cfg['TASK']
        video_root = cfg['VIDEO_ROOT'](task)
        video_codes_save = cfg['VIDEO_CODES_SAVE'](task)
        video_recon_save = cfg['VIDEO_RECON_SAVE']
    else:
        video_root = cfg['VIDEO_ROOT']
        video_codes_save = cfg['VIDEO_CODES_SAVE']
        video_recon_save = cfg['VIDEO_RECON_SAVE']

    return {
        'interval': interval,
        'min_pixels': min_pixels,
        'SIZE': size,
        'hz': hz,
        'VIDEO_ROOT': video_root,
        'VIDEO_CODES_SAVE': video_codes_save,
        'VIDEO_RECON_SAVE': video_recon_save
    }

if __name__ == "__main__":

    path = "/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/ckpt/Emu3-VisionVQ"

    # choose the dataset to process
    process_data = 'libero'

    # current supported datasets
    simulator_list = ["Calvin", "Calvin_partial", "libero", 'libero_long', 'maniskill']
    oxe_list = ['RT1', 'DROID', 'BridgeV2', 'bridge_orig', 'fmb', 'toto', 'taco_play','kuka',\
                'berkeley_autolab_ur5','viola', 'cmu_play_fusion', 'utaustin_mutex']
    aloha_list = ['aloha_songling']
    video_list = ['1X', 'Ego5m', 'SSv2']

    assert process_data in simulator_list + oxe_list + video_list + aloha_list, f"Invalid process_data: {process_data}"

    model = AutoModel.from_pretrained(path, trust_remote_code=True).eval().cuda()
    processor = AutoImageProcessor.from_pretrained(path, trust_remote_code=True)
    
    # Retrieve configuration for the selected dataset
    config = get_data_config(process_data)

    # Assign configuration values to variables used in your pipeline
    processor.min_pixels = config['min_pixels']  # Minimum valid pixel count
    interval = config['interval']                # Frame interval (e.g., sampling rate)
    SIZE = config['SIZE']                        # Desired image resolution (width, height)
    hz = config['hz']                            # Final video frequency (Hz)

    # Paths for loading raw video, saving codes, and reconstructed videos
    VIDEO_ROOT = config['VIDEO_ROOT']
    VIDEO_CODES_SAVE = config['VIDEO_CODES_SAVE']
    VIDEO_RECON_SAVE = config['VIDEO_RECON_SAVE']

    os.makedirs(VIDEO_CODES_SAVE, exist_ok=True)
    os.makedirs(VIDEO_RECON_SAVE, exist_ok=True)

    try:
        rank = int(sys.argv[1])
    except Exception as e:
        print(f"Error parsing rank: {e}")
    videos = sorted(os.listdir(VIDEO_ROOT))[rank::8]
    # videos = sorted(os.listdir(VIDEO_ROOT))[rank::4]

    single_view_datasets = ['RT1', 'BridgeV2', 'Ego5m', 'maniskill', 'fmb', 'toto',\
         'kuka', 'viola', 'utaustin_mutex', 'cmu_play_fusion', 'SSv2']
    multi_view_datasets = ['taco_play','aloha','bridge_orig', 'berkeley_autolab_ur5', 'aloha_pour','aloha_songling',\
         'aloha_8task', 'aloha_fold']
    
    for video in tqdm(videos, desc="Processing videos"):
        print("processing videos: ", video)
        # RT1
        if process_data in single_view_datasets:
            images, image_paths = load_images(osp.join(VIDEO_ROOT, video,'images'), SIZE, interval)
        elif process_data == 'DROID' or process_data == '1X':
            images, image_paths = load_images(osp.join(VIDEO_ROOT, video), SIZE, interval)
        elif process_data == 'Calvin':
            # images, image_paths = load_images(osp.join(VIDEO_ROOT, video,'rgb_static'), SIZE, interval, augmentation=random_shift)
            images, image_paths = load_images(osp.join(VIDEO_ROOT, video,'rgb_gripper'), SIZE, interval, augmentation=random_shift)
            # images, image_paths = load_images(osp.join(VIDEO_ROOT, video,'rgb_gripper'), SIZE, interval, augmentation=random_brightness_enhance)
        elif process_data == 'Calvin_raw':
            images, image_paths = load_images(osp.join(VIDEO_ROOT, video), SIZE, interval)
        elif process_data == 'libero':
            images, image_paths = load_images(osp.join(VIDEO_ROOT, video, 'images'), SIZE, interval, augmentation=random_shift)
            # images, image_paths = load_images(osp.join(VIDEO_ROOT, video,'gripper_images'), SIZE, interval, augmentation=random_shift)
        elif process_data == 'Calvin_partial':
            images, image_paths = load_images(osp.join(VIDEO_ROOT, video,'rgb_static'), SIZE, interval)
            # images, image_paths = load_images(osp.join(VIDEO_ROOT, video,'rgb_gripper'), SIZE, interval)
        elif process_data == 'taco_play':
            views = ['images', 'grippers']
            for view in views:
                view_path = osp.join(VIDEO_ROOT, video, view)
                images, image_paths = load_images(view_path, SIZE, interval)
                processed_codes_path = osp.join(VIDEO_CODES_SAVE, video, view)
                recon_images_path = osp.join(VIDEO_RECON_SAVE, video, view)

                if os.path.exists(processed_codes_path):
                    print(f"Skipping video {video} view {view} as it has already been processed.")
                    continue

                image_level_enc_dec(images, model, processor, processed_codes_path, recon_images_path) 
        elif process_data == 'berkeley_autolab_ur5':
            views = ['images', 'gripper']
            for view in views:
                view_path = osp.join(VIDEO_ROOT, video, view)
                images, image_paths = load_images(view_path, SIZE, interval)
                processed_codes_path = osp.join(VIDEO_CODES_SAVE, video, view)
                recon_images_path = osp.join(VIDEO_RECON_SAVE, video, view)

                if os.path.exists(processed_codes_path):
                    print(f"Skipping video {video} view {view} as it has already been processed.")
                    continue

                image_level_enc_dec(images, model, processor, processed_codes_path, recon_images_path)  
        elif process_data == 'bridge_orig':
            views = ['images', 'images1']
            for view in views:
                view_path = osp.join(VIDEO_ROOT, video, view)
                images, image_paths = load_images(view_path, SIZE, interval)
                processed_codes_path = osp.join(VIDEO_CODES_SAVE, video, view)
                recon_images_path = osp.join(VIDEO_RECON_SAVE, video, view)

                if os.path.exists(processed_codes_path):
                    print(f"Skipping video {video} view {view} as it has already been processed.")
                    continue

                image_level_enc_dec(images, model, processor, processed_codes_path, recon_images_path)   
        elif process_data == 'aloha':
            views = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
            for view in views:
                view_path = osp.join(VIDEO_ROOT, video, view)
                images, image_paths = load_images(view_path, SIZE, interval)
                processed_codes_path = osp.join(VIDEO_CODES_SAVE, video, view)
                recon_images_path = osp.join(VIDEO_RECON_SAVE, video, view)

                if os.path.exists(processed_codes_path):
                    print(f"Skipping video {video} view {view} as it has already been processed.")
                    continue

                image_level_enc_dec(images, model, processor, processed_codes_path, recon_images_path)
        elif process_data == 'aloha_pour' or process_data == 'aloha_songling' or process_data == 'aloha_8task' or process_data == 'aloha_fold':
            views = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
            for view in views:
                view_path = osp.join(VIDEO_ROOT, video, view)
                # images, image_paths = load_images(view_path, SIZE, interval, augmentation=random_brightness_enhance)
                images, image_paths = load_images(view_path, SIZE, interval)
                processed_codes_path = osp.join(VIDEO_CODES_SAVE, video, view)
                recon_images_path = osp.join(VIDEO_RECON_SAVE, video, view)

                if os.path.exists(processed_codes_path):
                    print(f"Skipping video {video} view {view} as it has already been processed.")
                    continue

                image_level_enc_dec(images, model, processor, processed_codes_path, recon_images_path)
        else:
            raise ValueError(f"Invalid process_data: {process_data}")
        
        if process_data not in multi_view_datasets:
            # Enumerate all images in the folder
            processed_codes_path = osp.join(VIDEO_CODES_SAVE, video) 
            recon_images_path = osp.join(VIDEO_RECON_SAVE, video)
            
            if os.path.exists(processed_codes_path):
                print(f"Skipping video {video} as it has already been processed.")
                continue
            # Clip-level encoding and decoding
            # clip_level_enc_dec(images_enumerated, model, processor, processed_codes_path, recon_images_path)
            
            # Image-level encoding and decoding for all enumerated images
            image_level_enc_dec(images, model, processor, processed_codes_path, recon_images_path)