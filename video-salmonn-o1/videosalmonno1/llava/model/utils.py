from transformers import AutoConfig
from llava.model import LlavaAVQwenForCausalLM, LlavaQwenForCausalLM
from llava.mm_utils import process_anyres_image, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoConfig, AutoTokenizer
import torch
import os
import json
from collections import OrderedDict
from peft import LoraConfig, get_peft_model
import time
from dataclasses import make_dataclass
import transformers

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

def load_qwen_lora_model(
    model_path,
    model_base=None,
    lora_enable=False,
    audio_visual=True,
    lora_llm_only=False,
    flash_attn=False,
    pretrain_weight=None,
    use_dora=False,
    load_full=False,
    lora_r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    **audio_config
):
    model_ckpt_path = model_path
    model_path = os.path.dirname(model_ckpt_path)

    lora_ckpt = os.path.join(model_path, "all_parameters.bin")

    with open(os.path.join(model_path, 'config.json'), 'r') as fp:
        config = json.load(fp)

    if model_base is None:
        model_base = config["_name_or_path"]
        while os.path.exists(os.path.join(model_base, "all_parameters.bin")):
            with open(os.path.join(model_base, 'config.json'), 'r') as fp:
                config = json.load(fp)
            model_base = config["_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    cfg_pretrained = AutoConfig.from_pretrained(model_base)
    model_args = config["model_args"]

    model_args["lora_r"] = lora_r
    model_args["lora_alpha"] = lora_alpha
    model_args["lora_dropout"] = lora_dropout


    TempData = make_dataclass('TempData', model_args)
    model_args = TempData(**model_args)

    overwrite_config = {"model_args": vars(model_args), "multi_frame_projector": model_args.multi_frame_projector, "multi_frame_num": model_args.multi_frame_num, "add_time_token": model_args.add_time_token, "use_mfcnn": model_args.use_mfcnn, "use_mftrans": model_args.use_mftrans, "use_flash_tower": model_args.use_flash_tower, "mf_split_init": model_args.mf_split_init}
    overwrite_config["model_args"]["segmentation"] = audio_config["segmentation"] if "segmentation" in audio_config else -1
    overwrite_config["model_args"]["do_rag"] = audio_config["do_rag"] if "do_rag" in audio_config else False
    overwrite_config["model_args"]["rag_input_frames"] = audio_config["rag_input_frames"] if "rag_input_frames" in audio_config else 1
    overwrite_config["model_args"]["rag_type"] = audio_config["rag_type"] if "rag_type" in audio_config else "direct"
    overwrite_config["model_args"]["rag_topk"] = audio_config["rag_topk"] if "rag_topk" in audio_config else 5

    for k, v in overwrite_config.items():
        setattr(cfg_pretrained, k, v)

    if audio_visual:
        model = LlavaAVQwenForCausalLM.from_pretrained(model_base, config=cfg_pretrained, cache_dir=None, attn_implementation="flash_attention_2", torch_dtype=(torch.bfloat16), **audio_config)
        model.get_model().initialize_vision_modules(model_args=model_args)
        model = model.to(torch.bfloat16) # .cuda()
    else:
        # model = LlavaQwenForCausalLM(cfg_pretrained, attn_implementation="flash_attention_2")
        model = LlavaQwenForCausalLM.from_pretrained(model_base, config=cfg_pretrained, cache_dir=None, attn_implementation="flash_attention_2", torch_dtype=(torch.bfloat16), **audio_config)
        model.get_model().initialize_vision_modules(model_args=model_args)
        # Do not load in float16
        model = model.to(torch.bfloat16) # .cuda()

    rag_flag = False
    if hasattr(model, "text_encoder"):
        text_encoder = model.text_encoder
        del model.text_encoder
        rag_flag = True

    if load_full and lora_enable:
        ckpt = torch.load(lora_ckpt, map_location='cpu')
        if model_ckpt_path != lora_ckpt:
            ckpt_2 = torch.load(model_ckpt_path, map_location='cpu')
            for k in ckpt_2.keys():
                ckpt[k] = ckpt_2[k]
        
        new_ckpt = OrderedDict()
        for k in ckpt.keys():
            if 'vision_tower' not in k:
                new_ckpt[k[len('module.'):]] = ckpt[k]
            else:
                # if not audio_visual:
                new_ckpt[k[len('module.'):]] = ckpt[k]

        kk = model.load_state_dict(new_ckpt, strict=False)
        print("Load full: ", len(kk.unexpected_keys), len(kk.missing_keys))

    if lora_enable:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=use_dora,
        )
        model.to(torch.bfloat16)
        if audio_visual:
            if True:
                speech_encoder = model.speech_encoder
                model.speech_encoder = None
                v_flag = False
                if hasattr(model.model, "vision_tower"):
                    vision_tower = model.model.vision_tower
                    del model.model.vision_tower
                    v_flag = True

                vidrag_flag = False
                if hasattr(model, "video_encoder"):
                    video_encoder = model.video_encoder
                    del model.video_encoder
                    vidrag_flag = True

                model = get_peft_model(model, lora_config)

                model.model.speech_encoder = speech_encoder
                if v_flag:
                    model.model.model.vision_tower = vision_tower
                if vidrag_flag:
                    model.model.video_encoder = video_encoder
        else:
            v_flag = False
            if hasattr(model.model, "vision_tower"):
                vision_tower = model.model.vision_tower
                del model.model.vision_tower
                v_flag = True

            model = get_peft_model(model, lora_config)
            
            if v_flag:
                model.model.model.vision_tower = vision_tower
    else:
        model.to(torch.bfloat16)

    if rag_flag:
        model.model.text_encoder = text_encoder
    
    if load_full and lora_enable:
        if pretrain_weight is not None and pretrain_weight != "None":
            ckpt = OrderedDict()
            if pretrain_weight is not None and pretrain_weight != "None":
                ckpt_3 = torch.load(pretrain_weight, map_location='cpu')
                for k in ckpt_3.keys():
                    if "speech" in k or "final_linear" in k:
                        key = k.replace("module.", "base_model.model.")
                        ckpt[key] = ckpt_3[k]
                print("Load Pretrain Weight")

            kk = model.load_state_dict(ckpt, strict=False)
            print(len(kk.unexpected_keys), len(kk.missing_keys))
            print(len([it for it in kk.missing_keys if 'lora' in it or 'final_linear' in it]))
    else:
        ckpt = torch.load(lora_ckpt, map_location='cpu')
        
        if pretrain_weight is not None and pretrain_weight != "None":
            ckpt_3 = torch.load(pretrain_weight, map_location='cpu')
            for k in ckpt_3.keys():
                key = k.replace("module.", "module.base_model.model.")
                ckpt[key] = ckpt_3[k]

        if model_ckpt_path != lora_ckpt:
            ckpt_2 = torch.load(model_ckpt_path, map_location='cpu')
            for k in ckpt_2.keys():
                ckpt[k] = ckpt_2[k]

        new_ckpt = OrderedDict()
        for k in ckpt.keys():
            if 'vision_tower' not in k:
                new_ckpt[k[len('module.'):]] = ckpt[k]
            else:
                # if not audio_visual:
                new_ckpt[k[len('module.'):]] = ckpt[k]

        kk = model.load_state_dict(new_ckpt, strict=False)

        print(len(kk.unexpected_keys), len(kk.missing_keys))
        print(len([it for it in kk.missing_keys if 'lora' in it or 'final_linear' in it]))

    return model, tokenizer

