import os
from config.config import Config
import argparse
import yaml
import json
from omegaconf import OmegaConf

from datasets import SupervisedAudioVisualDataset4Test
from model.openllama import OpenLLAMAPEFTModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load arguments
args = parse_args()
args = Config(args).config

# Load the list of test files
all_decode_info = args.all_decode_info

# Set the decoder output directory
decode_root = os.path.dirname(args.delta_ckpt_path)
current_time = datetime.now()
timestamp = current_time.strftime("%Y%m%d%H%M")
decode_root = os.path.join(decode_root, timestamp)
os.makedirs(decode_root, exist_ok=True)
OmegaConf.save(args, os.path.join(decode_root, "config.yaml"))

# Initialise the model
ds_engine = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
ds_engine.load_state_dict(delta_ckpt, strict=False)
ds_engine = ds_engine.eval().half().to(device)

# Load test data as a list of dataloaders
dataloader_lst = []
for modality, task, data_path in all_decode_info:
    print("Loading data from: {}".format(data_path))

    if modality == "audio":
        dataset = SupervisedAudioVisualDataset4Test(
            'audio',
            audio_data_path=data_path,
            use_whisper=args["use_whisper"],
            training=False,
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"],
            cache_dir=args["cache_dir"],
        )
    elif modality == "audioimage":
        dataset = SupervisedAudioVisualDataset4Test(
            'audioimage',
            audio_data_path="./dummy/dummy_audio.json",
            image_data_path=data_path,
            use_whisper=args["use_whisper"],
            training=False,
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"],
            cache_dir=args["cache_dir"],
        )
    elif modality == "audiovideoimage":
        dataset = SupervisedAudioVisualDataset4Test(
            'audiovideoimage',
            audio_data_path="./dummy/dummy_audio.json",
            video_data_path=data_path,
            use_whisper=args["use_whisper"],
            training=False,
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"],
            cache_dir=args["cache_dir"],
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args['batch_size'],
        num_workers=3,
        shuffle=False,
        collate_fn=dataset.collate,
        drop_last=False
    )
   
    dataloader_lst.append([dataloader, task])

# Start inference
results = []
pbar = tqdm(total=sum([len(dataloader) for dataloader, _ in dataloader_lst]), desc="Decoding", position=0)

for dataloader, task in dataloader_lst:
    for batch_i, batch in enumerate(dataloader):
        with torch.no_grad():
            text = ds_engine(batch, generate=True)
            print(text)
            for gen, ref, id in zip(text, batch['output_texts'], batch['orig_paths']):
                results.append(
                    {
                        "id": f"{str(id)}_{ref[0]['value']}",
                        "conversation": ref,
                        "task": task,
                        "ref_answer": ref[1]['value'],
                        "gen_answer": gen
                    }
                )
            pbar.update(1)

# Write the results out
with open(os.path.join(decode_root, f"eval_result.json"), "w", encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)