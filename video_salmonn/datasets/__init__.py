# from header import *
from torch.utils.data import Dataset, DataLoader
import torch
from .samplers import DistributedBatchSampler
from .sft_dataset import SupervisedAudioVisualDataset, SupervisedDataset
from .sft_dataset_nomix import SupervisedAudioVisualDataset4Test

'''
def get_tokenizer(model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    tokenizer.bos_token_id, tokenizer.eos_token_id = 1, 2
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
'''

def load_sft_dataset(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    if args["data_type"] == "video":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            video_data_path=args['data_path'],
            video_root_path=args['image_root_path'],
        )
    elif args["data_type"] == "image":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            image_data_path=args['image_data_path'],
            image_root_path=args['llava_root_path'],
        )
    elif args["data_type"] == "videoimage":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            video_data_path=args['data_path'],
            video_root_path=args['image_root_path'],
            image_data_path=args['image_data_path'],
            image_root_path=args['llava_root_path'],
        )
    elif args["data_type"] == "audio":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            audio_data_path=args['audio_data_path'],
            audio_root_path=args['image_root_path'],
            use_whisper=args["use_whisper"],
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"]
        )
    elif args["data_type"] == "audioimage":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            audio_data_path=args['audio_data_path'],
            audio_root_path=args['image_root_path'],
            image_data_path=args['image_data_path'],
            image_root_path=args['llava_root_path'],
            use_whisper=args["use_whisper"],
        )
    elif args["data_type"] == "audiovideoimage":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            audio_data_path=args['audio_data_path'],
            audio_root_path=args['image_root_path'],
            image_data_path=args['image_data_path'],
            image_root_path=args['llava_root_path'],
            video_data_path=args['data_path'],
            video_root_path=args['image_root_path'],
            use_whisper=args["use_whisper"],
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"],
            audio_only=args.get('audio_only', False),
            video_only=args.get('video_only', False),
            use_npy=args.get('use_npy', False),
        )
    else:
        data = SupervisedDataset(args['data_path'], args['image_root_path'])

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=3,
        collate_fn=data.collate, 
        pin_memory=True
    )
    return data, iter_, sampler


def load_sft_dataset_val(args, drop_last=True):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    if args["data_type"] == "video" or args["data_type"] == "videoimage":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            video_data_path=args['val_data_path'],
            video_root_path=args['image_root_path'],
            training=False,
        )
    elif args["data_type"] == "image":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            image_data_path=args['image_val_data_path'],
            image_root_path=args['llava_root_path'],
            training=False,
        )
    elif args["data_type"] == "audio":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            audio_data_path=args['audio_val_data_path'],
            audio_root_path=args['image_root_path'],
            use_whisper=args["use_whisper"],
            training=False,
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"]
        )
    elif args["data_type"] == "audioimage":
        data = SupervisedAudioVisualDataset(
            args['data_type'],
            audio_data_path=args['audio_val_data_path'],
            audio_root_path=args['image_root_path'],
            image_data_path=args['image_val_data_path'],
            image_root_path=args['llava_root_path'],
            use_whisper=args["use_whisper"],
            training=False,
        )
        # visualdata = SupervisedAudioVisualDataset(
        #     "image",
        #     image_data_path=args['image_val_data_path'],
        #     image_root_path=args['llava_root_path'],
        #     training=False,
        # )
        # data = [visualdata, avdata]

    elif args["data_type"] == "audiovideoimage":
        avdata = SupervisedAudioVisualDataset(
            args['data_type'],
            video_data_path=args['val_data_path'],
            video_root_path=args['image_root_path'],
            use_whisper=args["use_whisper"],
            training=False,
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"],
            video_only=args.get('video_only', False),
        )
        if args.get('video_only', False):
            data = avdata
        else:
            visualdata = SupervisedAudioVisualDataset(
                "audioimage",
                audio_data_path=args['audio_val_data_path'],
                audio_root_path=args['image_root_path'],
                image_data_path=args['image_val_data_path'],
                image_root_path=args['llava_root_path'],
                use_whisper=args["use_whisper"],
                training=False,
                sin_pos=args["sin_pos"],
                return_raw=args["return_raw"]
            )
            data = [visualdata, avdata]
    else:
        data = SupervisedDataset(args['data_path'], args['image_root_path'])

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        drop_last,
        rank=rank,
        world_size=world_size
    )
    if isinstance(data, list):
        audio_sampler = torch.utils.data.RandomSampler(avdata)
        audio_batch_sampler = DistributedBatchSampler(
            audio_sampler, 
            batch_size,
            True,
            rank,
            world_size
        )
        audio_iter_ = DataLoader(
            avdata, 
            batch_sampler=audio_batch_sampler, 
            num_workers=0,
            collate_fn=avdata.collate, 
            pin_memory=True
        )
        video_sampler = torch.utils.data.RandomSampler(visualdata)
        video_batch_sampler = DistributedBatchSampler(
            video_sampler, 
            batch_size,
            True,
            rank,
            world_size
        )
        image_iter_ = DataLoader(
            visualdata, 
            batch_sampler=video_batch_sampler, 
            num_workers=0,
            collate_fn=visualdata.collate, 
            pin_memory=True
        )
        iter_ = [image_iter_, audio_iter_]
    else:
        iter_ = DataLoader(
            data, 
            batch_sampler=batch_sampler, 
            num_workers=4,
            collate_fn=data.collate, 
            pin_memory=True
        )
    return data, iter_, sampler