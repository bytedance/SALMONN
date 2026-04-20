# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2026 Tsinghua University (Siyin Wang)
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
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from ..utils.common import IGNORE_ID
from ..transformer.label_smoothing_loss import LabelSmoothingLoss
from ..utils.common import th_accuracy
from ..utils.file_utils import logging
import math

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 50 # 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids
    
    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        if self.fp16 is True:
            embedding = embedding.half()

        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class Qwen2LM(TransformerLM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. streaming strategy
        self.text_chunk = 4#2#1#4
        self.audio_chunk = 15#8#4#15
        self.ratio = self.audio_chunk / self.text_chunk
        self.only_streaming_training = False
    
    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token_seq, text_token_len, task_id_emb, speech_token_seq, speech_token_len):
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token_seq[i], task_id_emb.squeeze(dim=0), speech_token_seq[i]], dim=0)
                    for i in range(len(text_token_seq))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=0)
        return lm_input, lm_input_len

    def stream_pad_unpad_sequence(self, sos_eos_emb, embedding, text_token_seq, text_token_len, task_id_emb, speech_token_seq, speech_token_len, segments, prompt_text_token):
        lm_input = []
        for i in range(len(text_token_seq)):
            input = [sos_eos_emb.squeeze(dim=0), embedding[i]]
            if len(prompt_text_token) > 0:
                if len(prompt_text_token[i]) > 0:
                    input = input + [prompt_text_token[i]] 
            text_segments, audio_segments, audio_pos = segments[i]
            for text_segment, audio_segment in zip(text_segments, audio_segments):
                input = input + [text_token_seq[i][text_segment[0] : text_segment[1]]] \
                              + [speech_token_seq[i][audio_segment[0] : audio_segment[1]]]
            input = input + [task_id_emb.squeeze(dim=0), speech_token_seq[i][audio_pos:]]
            lm_input.append(torch.concat(input, dim=0))

        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=0)
        return lm_input, lm_input_len

    def interleave_tokens(self, text_len, audio_len, text_chunk, audio_chunk):
        ratio = self.ratio
        if text_len < text_chunk:
            text_chunk, audio_chunk = text_len, int(text_len * ratio)
        if audio_len / text_len < self.ratio:
            ratio = audio_len / text_len
            audio_chunk = math.floor(text_chunk * ratio) - 1

        result = []
        text_segments, audio_segments = [], []
        text_pos, audio_pos = 0, 0
        while text_len >= text_chunk and audio_len >= audio_chunk:
            text_segments.append((text_pos, text_pos+text_chunk))
            audio_segments.append((audio_pos, audio_pos+audio_chunk))
            text_pos, audio_pos = text_pos+text_chunk, audio_pos+audio_chunk
            text_len -= text_chunk
            audio_len -= audio_chunk

        if text_len > 0 and audio_len > 0:
            last_audio_len = int(text_len * ratio)
            text_segments.append((text_pos, text_pos+text_len))
            audio_segments.append((audio_pos, audio_pos+last_audio_len))
            text_pos, audio_pos = text_pos+text_len, audio_pos+last_audio_len

        return text_segments, audio_segments, audio_pos
    
    def forward(
        self,
        text_token,
        text_token_len,
        speech_token,
        speech_token_len,
        prompt_text_token = []
        # device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L)
            text_lengths: (B,)
            audio: (B, T)
            audio_lengths: (B,)
        """
        device = text_token.device
        batch_size = text_token.shape[0]

        # 1a. prepare lm_target
        if not self.only_streaming_training:
            lm_target = [torch.tensor([IGNORE_ID] * (1 + text_token_len[i]) + speech_token[i, :speech_token_len[i].item()].tolist() +
                                [self.speech_token_size]) for i in range(text_token.size(0))]
            lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1b. prepare streaming lm_target
        # text_segments =  [(0, 20), (20, 50)]
        # audio_segments = [(0, 75), (75, 188)]
        # audio_pos = 188
        stream_lm_target = []
        segments = []
        for i in range(batch_size):
            text_segments, audio_segments, audio_pos = self.interleave_tokens(text_token_len[i].item(), speech_token_len[i].item(),
                                                                              self.text_chunk, self.audio_chunk)
            segments.append((text_segments, audio_segments, audio_pos))
            target = []
            if len(prompt_text_token) > 0:
                if len(prompt_text_token[i]) > 0:
                    target = target + [IGNORE_ID] * (prompt_text_token[i].shape[0])
            for text_segment, audio_segment in zip(text_segments, audio_segments):
                target = target + [IGNORE_ID] * (text_segment[1] - text_segment[0]) \
                                + speech_token[i, audio_segment[0] : audio_segment[1]].tolist()
            target = target + [IGNORE_ID] + speech_token[i, audio_pos : speech_token_len[i].item()].tolist() + [self.speech_token_size]
            stream_lm_target.append(torch.tensor(target))

        stream_lm_target = pad_sequence(stream_lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)
        if self.only_streaming_training:
            lm_target = stream_lm_target
        else:
            lm_target = torch.cat([lm_target, stream_lm_target], dim=0)

        # 2. embedding projection
        embedding = torch.zeros(batch_size, 0, self.llm_input_size).to(device)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        text_token_seq = unpad_sequence(text_token, text_token_len, batch_first=True)
        speech_token_seq = unpad_sequence(speech_token, speech_token_len, batch_first=True)
        stream_lm_input, stream_lm_input_len = self.stream_pad_unpad_sequence(sos_eos_emb, embedding, text_token_seq, text_token_len,
                                                                    task_id_emb, speech_token_seq, speech_token_len, segments, prompt_text_token)
        if self.only_streaming_training:
            lm_input = stream_lm_input
            lm_input_len = stream_lm_input_len
        else:
            lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token_seq, text_token_len,
                                                            task_id_emb, speech_token_seq, speech_token_len)
            lm_input = torch.cat([lm_input, stream_lm_input], dim=0)
            lm_input_len = torch.cat([lm_input_len, stream_lm_input_len], dim=0)

        # 6. run lm forward
        new_bs, new_seqlen, _ = lm_input.shape
        attention_mask = torch.zeros(new_bs, new_seqlen, dtype=torch.long, device=device)
        for i, length in enumerate(lm_input_len):
            attention_mask[i, :length] = 1

        output = self.llm.model(inputs_embeds=lm_input, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

        logits = self.llm_decoder(output.hidden_states[-1])
        loss = self.criterion_ce(logits, lm_target)
        # loss = self.ce_loss(logits.view(-1, logits.size(-1)), lm_target.view(-1))
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)
        
        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    @torch.inference_mode()
    def inference_salmonn(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
            prompt_text_len: torch.Tensor = torch.tensor([0], dtype=torch.int32),
            prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_token_len: torch.Tensor = torch.tensor([0], dtype=torch.int32),
            embedding: torch.Tensor = torch.zeros(0, 192),
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        prompt_text_len = prompt_text_len.to(device)
        if prompt_text_len != 0:
            text = torch.concat([prompt_text, text], dim=1)
            text_len += prompt_text_len
        
        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            # yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        return out_tokens

    def inference_salmonn_stream_teacherforcing(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
            prompt_text_len: torch.Tensor = torch.tensor([0], dtype=torch.int32),
            prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_token_len: torch.Tensor = torch.tensor([0], dtype=torch.int32),
            embedding: torch.Tensor = torch.zeros(0, 192),
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        prompt_t, prompt_s, t = prompt_text_len.item(), prompt_speech_token_len.item(), text_len.item()
        prompt_text_len = prompt_text_len.to(device)
        if prompt_text_len != 0:
            text = torch.concat([prompt_text, text], dim=1)
            text_len += prompt_text_len
        
        # 1. arrange streaming segments
        text_segments =  [(0, prompt_t)]
        audio_segments = [(0, prompt_s)]
        num_segments = t // self.text_chunk
        if num_segments == 0: num_segments = 1
        for _ in range(num_segments):
            text_segments.append((prompt_t, prompt_t + self.text_chunk))
            audio_segments.append((prompt_s, prompt_s + self.audio_chunk))
            prompt_t, prompt_s = prompt_t + self.text_chunk, prompt_s + self.audio_chunk
        text_segments.append((prompt_t, text_len.item()))
        audio_segments.append((prompt_s, prompt_s + int((text_len.item() - prompt_t) * self.ratio)))
        # text_segments =  [(0, 22),  (22, 42),   (42, 62),   (62, 92)]
        # audio_segments = [(0, 195), (195, 270), (270, 345), (345, 457)]

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        lm_input = torch.concat([sos_eos_emb, embedding, text[:, text_segments[0][0] : text_segments[0][1], :], 
                         prompt_speech_token_emb[:, audio_segments[0][0] : audio_segments[0][1], :]], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. streaming tts decoding
        out_tokens = []

        for idx in range(1, len(text_segments)):
            text_segment, audio_segment = text_segments[idx], audio_segments[idx]

            cache = None
            lm_input = torch.cat([lm_input, text[:, text_segment[0] : text_segment[1], :]], dim=1)
            step_lm_input = lm_input
            for i in range(audio_segment[1] - audio_segment[0]):
                lastcache = cache
                y_pred, cache = self.llm.forward_one_step(step_lm_input,
                                                        masks=torch.tril(torch.ones((1, step_lm_input.shape[1], step_lm_input.shape[1]), device=step_lm_input.device)).to(torch.bool),
                                                        cache=cache)
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                if top_ids > self.speech_token_size: 
                    cache = lastcache
                    continue
                # yield top_ids
                out_tokens.append(top_ids)
                step_lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
                lm_input = torch.cat([lm_input, step_lm_input], dim=1)

        lm_input = torch.cat([lm_input, task_id_emb], dim=1)
        cache = None
        already_decoded = audio_segments[-1][1] - audio_segments[0][1]
        for i in range(max_len - already_decoded):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            # yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        
        return out_tokens
    
    def inference_salmonn_stream(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            token_input,
            text_idx_num: int = 0,  
            generated: list = [],
            prompt_text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
            prompt_text_len: torch.Tensor = torch.tensor([0], dtype=torch.int32),
            prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_token_len: torch.Tensor = torch.tensor([0], dtype=torch.int32),
            embedding: torch.Tensor = torch.zeros(0, 192),
            sampling: int = 25,
            max_token_text_ratio: float = 15,
            min_token_text_ratio: float = 2,
            static_sent: bool = True,
    ):
        device = text.device
        prompt_text_len = prompt_text_len.to(device)

        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()

        if token_input is None:
            prompt_t, prompt_s, t = prompt_text_len.item(), prompt_speech_token_len.item(), text_len.item()
            if prompt_text_len != 0:
                text = torch.concat([prompt_text, text], dim=1)
                text_len += prompt_text_len
            
            # 1. arrange streaming segments
            text_segments =  [(0, prompt_t)]
            audio_segments = [(0, prompt_s)]

            # 2. encode embedding
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

            # 3. concat llm_input
            sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
            task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
            if prompt_speech_token_len != 0:
                prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
            else:
                prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

            lm_input = torch.concat([sos_eos_emb, embedding, text[:, text_segments[0][0] : text_segments[0][1], :], prompt_speech_token_emb[:, audio_segments[0][0] : audio_segments[0][1], :]], dim=1)

            # 4. cal min/max_length
            min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
            max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

            # 5. streaming tts decoding
            cur_generated = []
            generated = []
            final = False

            cache = None
            lm_input = torch.cat([lm_input, text[:, text_idx_num + prompt_text_len: self.text_chunk + prompt_text_len, :]], dim=1)
            step_lm_input = lm_input
            for i in range(self.audio_chunk):
                lastcache = cache
                y_pred, cache = self.llm.forward_one_step(step_lm_input,
                                                        masks=torch.tril(torch.ones((1, step_lm_input.shape[1], step_lm_input.shape[1]), device=step_lm_input.device)).to(torch.bool),
                                                        cache=cache)
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                logp = logp[:, :self.speech_token_size]
                top_ids = self.sampling_ids(logp.squeeze(dim=0), generated, sampling, ignore_eos=True if i < min_len else False).item()
                if top_ids > self.speech_token_size: 
                    cache = lastcache
                    continue
                # yield top_ids
                cur_generated.append(top_ids)
                generated.append(top_ids)
                step_lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
                lm_input = torch.cat([lm_input, step_lm_input], dim=1)
        
        else:
            min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
            max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
            if text_idx_num < text_len.item():
                text_todo_len = min(self.text_chunk, text_len.item() - text_idx_num)
                audio_todo_len = int(min(self.audio_chunk, self.ratio * text_todo_len))
                cur_generated = []
                final = False

                cache = None
                lm_input = torch.cat([token_input, text[:, text_idx_num : text_idx_num + text_todo_len, :]], dim=1)
                step_lm_input = lm_input
                
                for i in range(audio_todo_len):
                    lastcache = cache
                    y_pred, cache = self.llm.forward_one_step(step_lm_input,
                                                            masks=torch.tril(torch.ones((1, step_lm_input.shape[1], step_lm_input.shape[1]), device=step_lm_input.device)).to(torch.bool),
                                                            cache=cache)
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    logp = logp[:, :self.speech_token_size]
                    top_ids = self.sampling_ids(logp.squeeze(dim=0), generated, sampling, ignore_eos=True if i < min_len else False).item()
                    if top_ids > self.speech_token_size: 
                        cache = lastcache
                        continue
                    # yield top_ids
                    cur_generated.append(top_ids)
                    generated.append(top_ids)
                    step_lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
                    lm_input = torch.cat([lm_input, step_lm_input], dim=1)
                    # if len(generated) >= 300:
                    #     final = True
                    #     break
            else:
                task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

                lm_input = torch.cat([token_input, task_id_emb], dim=1)
                cache = None
                final = False
                cur_generated = []
                already_decoded = int(token_input.shape[1] * (1 - 1/(1 + self.ratio)))
                for i in range(max_len - already_decoded):
                    lastcache = cache
                    y_pred, cache = self.llm.forward_one_step(lm_input,
                                                            masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                            cache=cache)
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    # logp = logp[:, :self.speech_token_size + 1] # [Yu] remove filling tokens and etc.
                    top_ids = self.sampling_ids(logp.squeeze(dim=0), generated, sampling, ignore_eos=True if i < min_len else False).item()
                    # top_ids = self.sampling_ids(logp.squeeze(dim=0), generated, sampling, ignore_eos=False).item()
                    if top_ids == self.speech_token_size:
                        final = True
                        break
                    if top_ids > self.speech_token_size:
                        cache = lastcache
                        continue
                    # in stream mode, yield token one by one
                    # yield top_ids
                    cur_generated.append(top_ids)
                    generated.append(top_ids)
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
                    if i == max_len - already_decoded - 1: # or len(generated) >= 300:
                        final = True
                        break

        # ender.record()
        # torch.cuda.synchronize()
        # print(f"forward time: {starter.elapsed_time(ender)} ms")
        # import pdb; pdb.set_trace()

        return lm_input, generated, cur_generated, final

    @torch.inference_mode()
    def inference_bistream(
            self,
            text: Generator,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:

        device = prompt_text.device
        # 1. prepare input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=prompt_text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb], dim=1)

        # 2. iterate text
        out_tokens = []
        cache = None
        # NOTE init prompt_text as text_cache as it is basically impossible prompt_speech_token/prompt_text < 15/5
        text_cache = self.llm.model.model.embed_tokens(prompt_text)
        next_fill_index = -1
        for this_text in text:
            text_cache = torch.concat([text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1)
            # prompt_speech_token_emb not empty, try append to lm_input
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = text_cache[:, :self.mix_ratio[0]], prompt_speech_token_emb[:, :self.mix_ratio[1]]
                    logging.info('append {} text token {} speech token'.format(lm_input_text.size(1), lm_input_speech.size(1)))
                    lm_input = torch.concat([lm_input, lm_input_text, lm_input_speech], dim=1)
                    text_cache, prompt_speech_token_emb = text_cache[:, self.mix_ratio[0]:], prompt_speech_token_emb[:, self.mix_ratio[1]:]
                else:
                    logging.info('not enough text token to decode, wait for more')
                    break
            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2) or (len(out_tokens) == 0 and lm_input.size(1) == 1):
                    logging.info('get fill token, need to append more text token')
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, :self.mix_ratio[0]]
                        logging.info('append {} text token'.format(lm_input_text.size(1)))
                        if len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0]:]
                    else:
                        logging.info('not enough text token to decode, wait for more')
                        continue
                while True:
                    seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
                    y_pred, cache = self.llm.forward_one_step(lm_input,
                                                masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                cache=cache)
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.speech_token_size + 2
                        next_fill_index += (self.mix_ratio[1] + 1)
                    else:
                        top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True).item()
                    if top_ids == self.speech_token_size + 2:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info('fill_token index {} next fill_token index {}'.format(len(out_tokens), next_fill_index))
                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.speech_token_size + 2:
                            break
                        else:
                            raise ValueError('should not get token {}'.format(top_ids))
                    yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # 3. final decode
        lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1)
        logging.info('no more text token, decode until met eos')
        while True:
            seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False).item()
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.speech_token_size:
                    break
                else:
                    raise ValueError('should not get token {}'.format(top_ids))
            # in stream mode, yield token one by one
            yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
