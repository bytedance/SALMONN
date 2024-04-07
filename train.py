import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import *
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # load config
    cfg = Config(parse_args())
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # print config
    cfg.pretty_print()

    # build model
    model = load_model(model_config)

    # build datasets
    datasets = {
        "train": SALMONNDataset(data_config.train_ann_path, data_config.whisper_path),
        "valid": SALMONNDataset(data_config.valid_ann_path, data_config.whisper_path),
        "test": SALMONNDataset(data_config.test_ann_path, data_config.whisper_path),
    }

    # build runner
    runner = Runner(cfg, model, datasets, job_id)

    # train
    runner.train()


if __name__ == "__main__":
    main()