from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq
import torch
import shutil

path_in = "Qwen2.5-VL-72B-Instruct"
path_to_save = "Qwen2.5-VL-72B-Instruct-Audio"

tokenizer = AutoTokenizer.from_pretrained(
    path_in,
    model_max_length=131072,
    padding_side="right",
    use_fast=False,
)

tokenizer.add_tokens(["<|audio_pad|>"])
# 
tokenizer.save_pretrained(path_to_save)

attn_implementation="flash_attention_2"

model = video_SALMONN2_plus.from_pretrained(
    path_in,
    attn_implementation=attn_implementation,
    torch_dtype=(torch.bfloat16),
)

model_id = "openai/whisper-large-v3"

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)

for k, v in model.audio.named_parameters():
    if k in whisper_model.model.encoder.state_dict() and v.shape == whisper_model.model.encoder.state_dict()[k].shape:
        v.data = whisper_model.model.encoder.state_dict()[k].data
    else:
        print(k)

model.audio.q_tokens.data.normal_(mean=0.0, std=0.02)

model.save_pretrained(path_to_save)

shutil.copy(
    f"{path_in}/chat_template.json",
    f"{path_to_save}/chat_template.json"
)

shutil.copy(
    f"{path_in}/preprocessor_config.json",
    f"{path_to_save}/preprocessor_config.json"
)