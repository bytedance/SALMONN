import os
from typing import Dict, List, Optional, Sequence, Union, Any
# from contextlib import contextmanager, nullcontext

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

import os
import re
import multiprocessing
import time
from tqdm import tqdm

# 假设这些是您项目中用于操作HDFS的函数和常量
# from your_hdfs_utils import hmkdir, hlist_files, hrm, hcopy, HDFS_BASE_PATH

def _upload_checkpoint_in_background(local_output_dir, hdfs_target_dir, save_total_limit):
    """
    此函数在一个独立的进程中运行，以避免阻塞主训练进程。
    它负责将本地checkpoint上传到HDFS，并管理历史checkpoint数量。

    Args:
        local_output_dir (str): 本地保存的checkpoint目录路径 (例如: './outputs/checkpoint-500')
        hdfs_target_dir (str): HDFS上的目标根目录 (例如: 'your_model_output_dir')
        save_total_limit (int): HDFS上希望保留的checkpoint最大数量
    """
    try:
        print(f"[BG_PROCESS] 后台进程启动，开始处理 checkpoint: {local_output_dir}")
        base_path = os.path.join(HDFS_BASE_PATH, hdfs_target_dir)
        
        # 1. 确保HDFS上的根目录存在
        hmkdir(base_path)

        # 2. 清理旧的 Checkpoint
        # 列出所有已存在的checkpoint目录
        files = hlist_files([base_path])
        
        # 只有当远程文件数量已经达到或超过限制时，才进行清理
        # +1 是因为我们马上要上传一个新的
        if len(files) >= save_total_limit:
            steps_all = []
            pattern = r"(?<=checkpoint-)\d+"
            for file_path in files:
                # 从路径中提取文件名进行匹配
                file_name = os.path.basename(file_path)
                match = re.search(pattern, file_name)
                if match:
                    steps_all.append(int(match.group()))
            
            # 如果找到了符合规则的checkpoint
            if steps_all:
                steps_all.sort() # 从小到大排序
                
                # 计算需要删除的数量
                num_to_delete = len(steps_all) - save_total_limit + 1
                checkpoints_to_delete = steps_all[:num_to_delete]

                print(f"[BG_PROCESS] HDFS上已有 {len(steps_all)} 个 checkpoints，"
                      f"超过限制 {save_total_limit}。准备删除 {len(checkpoints_to_delete)} 个最旧的。")

                for step in checkpoints_to_delete:
                    path_to_delete = os.path.join(base_path, f"checkpoint-{step}")
                    print(f"[BG_PROCESS] 正在删除旧的 HDFS checkpoint: {path_to_delete}")
                    hrm(path_to_delete) # hrm 需要能处理目录删除

        # 3. 上传新的 Checkpoint
        # local_output_dir 是源，base_path 是目标父目录
        print(f"[BG_PROCESS] 开始从 {local_output_dir} 拷贝到 HDFS 目录 {base_path}")
        hcopy(local_output_dir, base_path)
        
        final_hdfs_path = os.path.join(base_path, os.path.basename(local_output_dir))
        print(f"[BG_PROCESS] Checkpoint 已成功保存到 HDFS: {final_hdfs_path}")

    except Exception as e:
        # 在后台进程中打印错误，避免静默失败
        print(f"[BG_PROCESS_ERROR] 后台上传checkpoint时发生错误: {e}")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, RandomSampler
from packaging import version
from transformers import Trainer
from transformers.cache_utils import Cache
from qwenvl.data.modality_sampler import WeightedRoundRobinBatchSampler
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.utils import (
    is_torch_compile_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    is_accelerate_available
)
from transformers.training_args import OptimizerNames

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

from transformers.trainer import (
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import torch.distributed as dist

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    SaveStrategy,
)

from transformers.trainer_callback import (
    ExportableState,
)

try:
    from cruise.utilities.distributed import DIST_ENV
    from cruise.utilities.hdfs_io import hcopy, hmkdir, hlist_files, hrm
except:
    pass

import re
from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOLoss
            

HDFS_BASE_PATH = "some HDFS Paths"  # Change to your own path

TRAINER_STATE_NAME = "trainer_state.json"


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
        "The final answer is:\n",
        "<answer>",
    ]
    for answer_prefix in answer_prefixes:
        s = s.split(answer_prefix)[-1]
        # s = s.replace(answer_prefix, "")
    if s == "":
        return s
    if s[0].lower() == s[0]:
        s = s[0].upper() + s[1:]
    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        return ""
    return matches[0]


def collate_fn(batch):
    return batch[0]


def _is_peft_model(model):
    # if is_peft_available():
    #     classes_to_check = (PeftModel,) if is_peft_available() else ()
    #     # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
    #     if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
    #         from peft import PeftMixedModel

    #         classes_to_check = (*classes_to_check, PeftMixedModel)
    #     return isinstance(model, classes_to_check)
    return False

class QwenVLTrainer(Trainer):

    def __init__(
        self, 
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dpo_loss_fct = LigerFusedLinearDPOLoss()

    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        # Build the sampler.
        if self.args.use_modality_sampler:
            return WeightedRoundRobinBatchSampler(
                train_dataset,
                self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size,
            )
        else:
            return RandomSampler(train_dataset)

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Run validation first
        val_loss = 0
        if self.args.do_validation:
            val_loss = self.validate()
            torch.distributed.reduce(val_loss, 0)
            val_loss = val_loss.item() / dist.get_world_size()
            if DIST_ENV.rank == 0:
                print("Validation Loss: {:.5f}".format(val_loss))

        self.save_model(output_dir, _internal_call=True)

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            # Wait for everyone to get here so we are sure the model has been saved by process 0
            # before we check if the best_checkpoint_dir exists
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()
                
            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        dist.barrier()
        if dist.get_rank() == 0:
            upload_process = multiprocessing.Process(
                target=_upload_checkpoint_in_background,
                args=(
                    output_dir,                  # 本地checkpoint的路径
                    self.args.output_dir,        # HDFS上的目标根目录
                    self.args.save_total_limit   # 保存限制
                )
            )
            # 启动进程
            upload_process.start()

    
    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters() if "merger" in name
                ]
                if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                    vision_tower_parameters = [
                        name for name, _ in opt_model.named_parameters() if "visual" in name
                    ]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def calc_dpo_loss(self, policy_input, policy_target, ref_input, ce_loss=None, beta=0.1):
        lm_head = self.model.lm_head.weight
        dpo_loss, (chosen_logp, reject_logp, chosen_logit, reject_logit, chosen_nll_loss, chosen_rewards, reject_rewards) = self.dpo_loss_fct(lm_head, policy_input, policy_target, ref_input=ref_input, ref_weight=lm_head)
        if ce_loss is not None:
            loss = dpo_loss + beta * ce_loss
        else:
            loss = dpo_loss
        print(f"RANK {dist.get_rank()} chosen: {chosen_rewards.item()}, reject: {reject_rewards.item()}")
        return (loss, dpo_loss, chosen_rewards, reject_rewards)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        train_type = inputs.get("train_type", "")
        if train_type == "sft":
            outputs = model(**inputs)
        elif train_type == "dpo":
            policy_input, policy_target = model(**inputs)
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                with torch.no_grad():
                    reference_input, reference_target = model(**inputs)
            outputs = self.calc_dpo_loss(policy_input, policy_target, reference_input)
            
        elif train_type == "gdpo":
            policy_input, policy_target, ce_loss = model(**inputs)
            inputs["train_type"] = "dpo"
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                with torch.no_grad():
                    reference_input, reference_target = model(**inputs)
            outputs = self.calc_dpo_loss(policy_input, policy_target, reference_input, ce_loss=ce_loss)
        else:
            raise NotImplementedError

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                if self.args.train_memory:
                    new_input = {k: v for k, v in inputs.items()}
                    new_input.pop("labels")
                    outputs = model(**new_input)
                    inputs["memory_triplets"] = outputs["memory_triplets"]
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available():
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    pass
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            # self.accelerator.backward(loss, **kwargs)
            model.backward(loss, **kwargs)
            model.optimizer.check_overflow()
            model.step()

            return loss.detach()

    def split_vision_chunks(self, text: str):
        # capture each block so re.split keeps the matches
        block = re.compile(
            r'(\s*<\d+(?:\.\d+)? seconds><\|vision_start\|>(?:<\|video_pad\|>)+(?:<\|audio_pad\|>)*<\|vision_end\|>)'
        )
        parts = [p for p in block.split(text) if p]  # keep non-empty pieces

        # Tidy: trim only the non-block outer pieces (start/end text)
        if parts and not block.fullmatch(parts[0]):
            parts[0] = parts[0].strip()
        if parts and not block.fullmatch(parts[-1]):
            parts[-1] = parts[-1].strip()
        return parts

    def prediction_step(
        self,
        model,
        inputs,
        do_sample=False,
        beam=1,
        prompts=None,
        processor=None,
    ):
        preds = []
        actions = []
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, max_new_tokens=512, do_sample=do_sample, top_p=0.9, temperature=1.0)
            preds = processor.decode(generated_tokens[0][inputs["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return preds

    def validate(self):
        model = self.model
        num_workers = 2 if self.args.video_max_frames > 512 else 8
        test_dataloader = DataLoader(self.eval_dataset, batch_size=1, collate_fn=collate_fn, num_workers=num_workers)
        test_dataloader = self.accelerator.prepare(test_dataloader)
        model.eval()
        total_loss = 0
        total_tokens = 0
        total_hits = 0
        unused_fields = ['distill_pixel_values_videos', 'distill_video_grid_thw', 'distill_input_ids', 'distill_labels', 'distill_attention_mask']

        if dist.get_rank() == 0:
            for inputs in tqdm(test_dataloader):
                refanswer = inputs.pop("ref_answer", [None])[0]
                prompt = inputs.pop("prompts", [None])[0]
                for unused_field in unused_fields:
                    inputs.pop(unused_field, None)
                inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                preds = self.prediction_step(
                    model,
                    inputs,
                    do_sample=False,
                    beam=1,
                    prompts=prompt,
                    processor=self.tokenizer,
                )
                pred = extract_characters_regex(preds[0] if len(preds) > 0 else "")
                refanswer = extract_characters_regex(refanswer)
                pred_correctness = 1.0 if pred == refanswer else 0.0
                total_hits += pred_correctness
                total_tokens += 1
        else:
            for inputs in test_dataloader:
                refanswer = inputs.pop("ref_answer", [None])[0]
                prompt = inputs.pop("prompts", [None])[0]
                for unused_field in unused_fields:
                    inputs.pop(unused_field, None)
                inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                preds = self.prediction_step(
                    model,
                    inputs,
                    do_sample=False,
                    beam=1,
                    prompts=prompt,
                    processor=self.tokenizer,
                )
                pred = extract_characters_regex(preds[0] if len(preds) > 0 else "")
                refanswer = extract_characters_regex(refanswer)
                pred_correctness = 1.0 if pred == refanswer else 0.0
                total_hits += pred_correctness
                total_tokens += 1
        return torch.tensor(total_hits/total_tokens*100).to(inputs["input_ids"].device)