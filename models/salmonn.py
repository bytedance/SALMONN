# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, StoppingCriteriaList, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

from .Qformer import BertConfig, BertLMHeadModel
from .modeling_llama import LlamaForCausalLM
from .modeling_whisper import WhisperModel
from .beats.BEATs import BEATsConfig, BEATs
from .utils import StoppingCriteriaSub


class PromptPool(nn.Module):
    def __init__(self, num_prompts=20, prompt_dim=1024, model_input_embeds=None):
        super().__init__()
        self.num_prompts = num_prompts
        self.usage_counts = torch.zeros(num_prompts, dtype=torch.long)  # Track frequency of each prompt

        
        # Initialize keys with small random values
        self.prompt_keys = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.01)  # Learnable keys
        
        # Initialize values based on model input embeddings mean plus noise
        if model_input_embeds is not None:
            self.prompt_values = nn.Parameter(model_input_embeds)
        else:
            self.prompt_values = nn.Parameter(torch.randn(num_prompts, prompt_dim))
            

    def forward(self, input_embedding, top_k=5):
        """
        Selects the top-k relevant prompts based on similarity with the input.
        Arguments:
        - input_embedding: [batch_size, hidden_dim]
        - top_k: Number of prompts to select
        """
        # Compute similarities between input and prompt keys
        similarities = torch.matmul(input_embedding, self.prompt_keys.T)  # [batch_size, num_prompts]

        topk_indices = torch.topk(similarities, top_k, dim=1).indices  # Get top-k prompt indices
        unique, counts = torch.unique(topk_indices, return_counts=True)
        self.usage_counts.index_add_(0, unique, counts)

        # Gather the selected prompts in a vectorized manner
        assert (topk_indices >= 0).all() and (topk_indices < self.prompt_values.size(0)).all(), "Index out of bounds!"

        selected_prompts = self.prompt_values[topk_indices]  # [batch_size, top_k, prompt_dim]
        return selected_prompts
    
    
class SALMONN(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        llama_model_name="",
        whisper_path="",
        freeze_whisper=True,
        beats_path="",
        freeze_beats=True,
        cache_dir = "",

        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        
        soft_prompts=True,
        num_soft_prompt_tokens=20,
        
        l2p=False,
        pool_size=30,
        prompt_size=10,

        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.beats_path = beats_path
        self.use_speech_Qformer = use_speech_Qformer
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride
        self.lora = lora
        self.multi_prompt = multi_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource
        
        self.use_soft_prompting = soft_prompts
        self.num_soft_prompt_tokens = num_soft_prompt_tokens
        
        self.l2p=l2p
        self.pool_size=pool_size
        self.prompt_size=prompt_size
        
        print(pool_size, type(pool_size))

        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            llama_model_name,
            cache_dir=cache_dir
        )
        
        logging.info('Loading LLaMA Model')
        if self.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit}
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16
            )
            
        if self.llama_tokenizer.pad_token is None:
            logging.info("There is no pad token present, using eos token instead")
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
      
        self.llama_tokenizer.padding_side = "right"
            
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLaMA Done')

        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()
            logging.info('LoRA Training')
            num_trainable_params = sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.llama_model.parameters())

        elif self.use_soft_prompting:
            logging.info("Using Soft Prompting for fine-tuning.")
            self.soft_prompt_length = self.num_soft_prompt_tokens
            with torch.no_grad():
                base_embedding_mean = self.llama_model.model.embed_tokens.weight.mean(dim=0)  # Compute mean embedding
                noise = torch.randn(1, self.num_soft_prompt_tokens, self.llama_model.config.hidden_size) * 0.02  # Small noise
                self.soft_prompt_embeddings = nn.Parameter(base_embedding_mean.unsqueeze(0).unsqueeze(0) + noise)
            num_trainable_params = self.soft_prompt_embeddings.numel()
            total_params = sum(p.numel() for p in self.llama_model.parameters()) + num_trainable_params
        
        elif self.l2p:
            logging.info("Using Learning to prompt for fine-tuning.")
            with torch.no_grad():
                base_embedding_mean = self.llama_model.model.embed_tokens.weight.mean(dim=0)  # Compute mean embedding
                print(self.pool_size, type(self.pool_size), self.llama_model.config.hidden_size, type(self.llama_model.config.hidden_size))
                noise = torch.randn(self.pool_size, self.llama_model.config.hidden_size) * 0.02  # Small noise
                self.prompt_pool = PromptPool(
                    num_prompts=self.pool_size,
                    prompt_dim=base_embedding_mean.shape[-1],
                    model_input_embeds=base_embedding_mean.unsqueeze(0) + noise)
            num_trainable_params = self.pool_size * base_embedding_mean.shape[-1]
            total_params = sum(p.numel() for p in self.llama_model.parameters()) + num_trainable_params

        # Compute percentage
        trainable_ratio = (num_trainable_params / total_params) * 100

        # Print results
        logging.info(f"Trainable params = {num_trainable_params:,} Total Params: {total_params:,} ({trainable_ratio:.4f}% of total params)")

        assert whisper_path
        logging.info('Loading Whisper Model')
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        if freeze_whisper:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            logging.info("freeze Whisper")
        
        if self.beats_path:
            logging.info("Loading BEATs Model")
            beats_ckpt = torch.load(self.beats_path, map_location='cpu')
            beats_cfg = BEATsConfig(beats_ckpt['cfg'])
            self.beats = BEATs(beats_cfg)
            self.beats.load_state_dict(beats_ckpt['model'])
            self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
            if freeze_beats:
                for name, param in self.beats.named_parameters():
                    param.requires_grad = False
                self.beats.eval()
                logging.info("freeze BEATs")

        if self.use_speech_Qformer:
            if self.beats_path:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim
                )
            else:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model
                )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                logging.info("freeze Speech QFormer")

            logging.info('Loading speech LLAMA proj')
            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            if speech_llama_proj_model:
                logging.info("Loading speech LLAMA proj from {}".format(speech_llama_proj_model))
                speech_llama_proj_weight = torch.load(speech_llama_proj_model, map_location="cpu")
                self.load_state_dict(speech_llama_proj_weight['model'], strict=False)
            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLAMA proj")
        else:
            # feel free to add other aligners here
            raise NotImplementedError

        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        with self.maybe_autocast():
            if self.use_speech_Qformer:
                speech_embeds = self.ln_speech(speech_embeds)
                if audio_embeds is not None:
                    audio_embeds = self.ln_audio(audio_embeds)
                    if audio_embeds.size(1) < speech_embeds.size(1):
                        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                    elif audio_embeds.size(1) > speech_embeds.size(1):
                        speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                    speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

                if self.window_level_Qformer:
                    B, T, C = speech_embeds.shape
                    kernel = round(1500 * self.second_per_window / 30.0)
                    stride = round(1500 * self.second_stride / 30.0)
                    kernel = (1, kernel)
                    stride = (1, stride)
                    speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                    speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                    _, _, L = speech_embeds_overlap.shape
                    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                    speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

                query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
                query_output = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_embeds,
                    encoder_attention_mask=speech_atts,
                    return_dict=True,
                )
                speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

                if self.window_level_Qformer:
                    speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            else:
                raise NotImplementedError

        return speech_embeds, speech_atts

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

            if self.beats_path and raw_wav is not None:
                audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            else:
                audio_embeds = None

        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)
    
    def inject_soft_prompt(self, inputs_embeds, input_representations=None):
        """
        Injects soft prompts into the input embeddings. Supports both fixed and L2P-style soft prompts.
        """
        if self.l2p:
            assert input_representations is not None, "Input representations are required for L2P."
            selected_prompts = self.prompt_pool(input_representations, top_k=self.prompt_size)  # Select relevant prompts
            inputs_embeds = torch.cat([selected_prompts, inputs_embeds], dim=1)
        else:
            batch_size = inputs_embeds.size(0)
            soft_prompts = self.soft_prompt_embeddings.expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)
        return inputs_embeds

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts

    def forward(self, samples, verbose=False):
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM
        text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        
        empty_target_size = speech_atts.shape[1] + 1 # speech tokens and bos token
        
        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)
        
        if self.use_soft_prompting or self.l2p:
            num_tokens = self.num_soft_prompt_tokens if self.use_soft_prompting else self.prompt_size
            inputs_embeds = self.inject_soft_prompt(inputs_embeds, speech_embeds.mean(1))
            soft_prompt_mask = torch.ones(
                inputs_embeds.shape[0], num_tokens, device=inputs_embeds.device, dtype=attention_mask.dtype
            )  # Create an attention mask for the soft prompts
            attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)
            empty_target_size += num_tokens
            
        # insert empty target tokens
        empty_targets = (
            torch.ones(
                [targets.shape[0], empty_target_size],
                dtype=torch.long
            ).to(spectrogram.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
            

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}

    def generate(self, samples, generate_cfg, prompts=None):
        batch_size = samples["spectrogram"].shape[0]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        if prompts is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)
        
        if self.use_soft_prompting:
            embeds = self.inject_soft_prompt(embeds, speech_embeds.mean(1))
            
            # Create a soft prompt attention mask (all ones)
            soft_prompt_mask = torch.ones(
                (embeds.shape[0], self.num_soft_prompt_tokens), 
                dtype=attns.dtype, 
                device=attns.device
            )
            # Prepend the soft prompt mask to the existing attention mask
            attns = torch.cat([soft_prompt_mask, attns], dim=1)

        stop_words_ids = [torch.tensor([2]).cuda()]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        return text

    @classmethod
    def from_config(cls, config):
        # llama_path = config.get("llama_path")
        llama_model_name = config.get("llama_model_name")
        cache_dir = config.get("cache_dir")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        beats_path = config.get("beats_path", "")
        freeze_beats = config.get("freeze_beats", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)
        
        soft_prompts = config.get("soft_prompts", False)
        num_soft_prompt_tokens = config.get("num_soft_prompt_tokens", 20)
        
        l2p = config.get("l2p", False)
        pool_size = config.get("pool_size", 30)
        prompt_size = config.get("prompt_size", 10)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        model = cls(
            llama_model_name=llama_model_name,
            cache_dir=cache_dir,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            soft_prompts=soft_prompts,
            num_soft_prompt_tokens=num_soft_prompt_tokens,
            l2p=l2p,
            pool_size=pool_size,
            prompt_size=prompt_size,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load SALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model
