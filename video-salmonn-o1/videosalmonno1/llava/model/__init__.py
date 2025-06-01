import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # Add other models as needed
    "llava_av_llama": "LlavaAVLlamaForCausalLM, LlavaAVConfig",
    "llava_av_qwen": "LlavaAVQwenForCausalLM, LlavaAVQwenConfig",
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        raise e
        print(f"Failed to import {model_name} from llava.language_model.{model_name}")
