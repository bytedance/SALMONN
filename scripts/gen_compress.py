from qwenvl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLPatchMerger
from transformers import AutoTokenizer
import torch
import shutil
import json

in_file = "/opt/tiger/thu_qwenvl/output/Qwen2.5-VL-7B-Instruct"
out_file = "/opt/tiger/thu_qwenvl/output/Qwen2.5-VL-7B-Instruct-F8"

scale_time = 4

tokenizer = AutoTokenizer.from_pretrained(
    in_file,
    model_max_length=131072,
    padding_side="right",
    use_fast=False,
)

tokenizer.save_pretrained(out_file)

attn_implementation="flash_attention_2"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    in_file,
    attn_implementation=attn_implementation,
    torch_dtype=(torch.bfloat16),
)

model.config.vision_config.spatial_merge_size *= scale_time

ori_merger = model.visual.merger

ori_dim = ori_merger.mlp[0].weight.shape[0]

with torch.no_grad():
    model.visual.merger = Qwen2_5_VLPatchMerger(
        dim=model.config.vision_config.out_hidden_size,
        context_dim=model.config.vision_config.hidden_size,
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
    )
    for i in range(scale_time):
        model.visual.merger.mlp[0].weight[i*ori_dim: (i+1)*ori_dim , i*ori_dim: (i+1)*ori_dim] = ori_merger.mlp[0].weight
        model.visual.merger.mlp[0].bias[i*ori_dim: (i+1)*ori_dim] = ori_merger.mlp[0].bias
        model.visual.merger.mlp[2].weight[:, i*ori_dim: (i+1)*ori_dim] = ori_merger.mlp[2].weight / scale_time

    model.visual.merger.mlp[2].bias = ori_merger.mlp[2].bias

    model.audio.q_tokens.data.normal_(mean=0.0, std=0.02)

model.save_pretrained(out_file)

shutil.copy(
    f"{in_file}/chat_template.json",
    f"{out_file}/chat_template.json"
)

with open(f"{in_file}/preprocessor_config.json", "r") as f:
    preprocessor_config = json.load(f)

# preprocessor_config["temporal_patch_size"] *= scale_time
preprocessor_config["merge_size"] *= scale_time

with open(f"{out_file}/preprocessor_config.json", "w") as f:
    json.dump(preprocessor_config, f, indent=2)

for k, v in model.named_parameters():
    if (v>100).any():
        print(k)