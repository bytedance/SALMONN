from lib2to3.pgen2.token import tok_name
import transformers
from .av_dataset import LazyAVSupervisedDataset, DataCollatorForAVSupervisedDataset, DataCollatorForAVSupervisedDatasetFullFrame
from .video_dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from .av_test_dataset import LazyAVTestDataset, DataCollatorForAVTestDataset
from .av_eval_dataset import LazyAVEvalDataset
from typing import Dict

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # if not data_args.use_audio: 
    #     train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    #     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # else:
    train_dataset = LazyAVSupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    if data_args.val_path is not None:
        test_dataset = LazyAVSupervisedDataset(tokenizer=tokenizer, data_path=data_args.val_path, data_args=data_args)
    else:
        test_dataset = train_dataset
    # if data_args.val_path is not None:
    #     test_dataset = LazyAVEvalDataset(tokenizer=tokenizer, data_path=data_args.val_path, data_args=data_args)
    # else:
    #     if data_args.val_ratio != 0.0:
    #         test_data = train_dataset.get_test_set()
    #         test_dataset = LazyAVEvalDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args, test_split=test_data)

    data_collator = DataCollatorForAVSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=data_collator)


def make_test_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for testing"""
    test_dataset = make_test_data(tokenizer, data_args)
    data_collator = DataCollatorForAVTestDataset(tokenizer=tokenizer)
    return dict(train_dataset=test_dataset, eval_dataset=test_dataset, data_collator=data_collator)

def make_test_data(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    # if not data_args.use_audio: 
    #     raise NotImplementedError()
    # else:
    test_dataset = LazyAVTestDataset(tokenizer=tokenizer, data_path=data_args.test_data_path, data_args=data_args)

    return test_dataset
