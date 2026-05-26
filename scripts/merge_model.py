from qwenvl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import torch

original_model = "/opt/tiger/thu_qwenvl/output/Qwen2.5-VL-7B-Instruct-Audio"

lora_ckpt = "/opt/tiger/thu_qwenvl/output/checkpoint-15475"

attn_implementation = "flash_attention_2"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    original_model,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

from peft import PeftModel
audio_layers = model.audio.layers
del model.audio.layers
model = PeftModel.from_pretrained(model, lora_ckpt)
model.model.audio.layers = audio_layers
model = model.merge_and_unload()

model.save_pretrained("/".join(lora_ckpt.split("/")[:-1]) + "/base")

print("/".join(lora_ckpt.split("/")[:-1]) + "/base")