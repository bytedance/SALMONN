import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BertLMHeadModelRaw

from .qformer import BertConfig
from .qformer import BertLMHeadModel as BertLMHeadModelQF

class VlmAttention(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        pretrain_qformer = getattr(model_args, "mm_vlmattention_pretrained", None)
        self.bert_type = getattr(model_args, "mm_vlmattention_bert_type", "qformer")
        self.num_query = getattr(model_args, "mm_vlmattention_num_query", 32)
        self.compress_type = getattr(model_args, "mm_vlmattention_compress_type", None)
        self.mm_hidden_size = self.hidden_size = vision_tower.hidden_size
        self.mm_vision_select_feature = model_args.mm_vision_select_feature
        self.language_hidden_size = 4096
        for_eval = True

        if 'pretrain' in self.bert_type:
            # for qformer that use evaclip for prtrain
            att_feat_size = 1408
        else:
            att_feat_size = self.mm_hidden_size
        self.vlm_att_tokenlizer, self.vlm_att_encoder, self.vlm_att_query = self.init_bert(att_feat_size, truncation_side="left")
        self.vlm_att_projector = torch.nn.Linear(self.vlm_att_encoder.config.hidden_size, self.mm_hidden_size)
        self.vlm_att_key_projector  = torch.nn.Linear(self.mm_hidden_size, self.mm_hidden_size)
        self.vlm_att_val_projector  = torch.nn.Linear(self.mm_hidden_size, self.language_hidden_size)

        if "raw" in self.bert_type:
            self.vlm_att_bert_proj  = torch.nn.Linear(att_feat_size, self.vlm_att_encoder.config.hidden_size)
        elif "pretrain" in self.bert_type and self.mm_hidden_size!=att_feat_size:
            self.vlm_att_bert_proj = torch.nn.Linear(self.mm_hidden_size, att_feat_size)
        else:
            self.vlm_att_bert_proj = None
        
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        
        if 'qformer_pretrain' in self.bert_type:
            self.vlm_att_ln = torch.nn.LayerNorm(att_feat_size)
        
        if pretrain_qformer is not None:
            print("Loading pretrained qformer weights...")
            qformer_weight = torch.load(pretrain_qformer, map_location='cpu')['model']
            bert_weight = {_key: qformer_weight[_key] for _key in qformer_weight if 'bert' in _key}
            self.vlm_att_encoder.load_state_dict(get_w(bert_weight, 'Qformer'))
            self.vlm_att_ln.load_state_dict(get_w(qformer_weight, 'ln_vision'))
            self.vlm_att_query.data = qformer_weight['query_tokens']
        
        if 'freeze_all' in self.bert_type:
            print("Freezing all qformer weights...")
            self.vlm_att_encoder.requires_grad_(False)
            self.vlm_att_ln.requires_grad_(False)
            self.vlm_att_query.requires_grad_(False)
            self.vlm_att_projector.requires_grad_(False)
            self.vlm_att_key_projector.requires_grad_(False)
            self.vlm_att_val_projector.requires_grad_(False)
        elif 'freeze' in self.bert_type:
            print("Freezing pretrained qformer weights...")
            self.vlm_att_encoder.requires_grad_(False)
            self.vlm_att_ln.requires_grad_(False)
            self.vlm_att_query.requires_grad_(False)
        

        if pretrain_mm_mlp_adapter is not None:
            att_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        else:
            trainable_module = ['vlm_att_encoder', 'vlm_att_projector', 'vlm_att_key_projector', 
                                'vlm_att_val_projector', 'vlm_att_query', 'vlm_att_visual_proj',
                                'vlm_att_ln']
            if hasattr(model_args, 'model_name_or_path'):
                model_save_path = model_args.model_name_or_path
            else:
                model_save_path = model_args.model_path
            model_idx_path = getattr(model_args, 'model_path', model_save_path)
            weight_file = json.load(open(os.path.join(model_idx_path, 'pytorch_model.bin.index.json'), 'r'))['weight_map']
            model_path = set([weight_file[_key] for _key in weight_file if any([_module in _key for _module in trainable_module])])
            att_projector_weights = {}
            for _model in model_path:
                att_projector_weights.update(torch.load(os.path.join(model_idx_path, _model), map_location='cpu'))
            if len(att_projector_weights) == 0:
                return
        
        bert_dict = get_w(att_projector_weights, 'vlm_att_encoder')
        if "bert.embeddings.position_ids" not in bert_dict and "raw_bert" not in self.bert_type:
            bert_dict["bert.embeddings.position_ids"] = self.vlm_att_encoder.bert.embeddings.position_ids
        print('Loading pretrained weights...')
        # import pdb;pdb.set_trace()

        self.vlm_att_encoder.load_state_dict(bert_dict)
        self.vlm_att_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_projector'))
        self.vlm_att_key_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_key_projector'))
        self.vlm_att_val_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_val_projector'))

        if "qformer" in self.bert_type:
            print('Loading vlm_att_query weights...')
            self.vlm_att_query.data = att_projector_weights['model.vlm_att_query']
            if "pretrain" in self.bert_type:
                print('Loading vlm_att_ln weights...')
                self.vlm_att_ln.load_state_dict(get_w(att_projector_weights, 'vlm_att_ln'))

        if self.vlm_att_bert_proj is not None:
            print('Loading vlm_att_bert_proj weights...')
            self.vlm_att_bert_proj.load_state_dict(get_w(att_projector_weights, 'vlm_att_bert_proj'))
        
        if for_eval:
            weight_type = torch.float16
            # import pdb;pdb.set_trace()
            # device_type = self.mm_projector[0].weight.device
            device_type = vision_tower.vision_tower.patch_embed.proj.weight.device
            self.vlm_att_encoder = self.vlm_att_encoder.to(device=device_type, dtype=weight_type)
            self.vlm_att_projector = self.vlm_att_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_key_projector = self.vlm_att_key_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_val_projector = self.vlm_att_val_projector.to(device=device_type, dtype=weight_type)

            if "qformer" in self.bert_type:
                self.vlm_att_query.data = self.vlm_att_query.data.to(device=device_type, dtype=weight_type)
                if "pretrain" in self.bert_type:
                    self.vlm_att_ln = self.vlm_att_ln.to(device=device_type, dtype=weight_type)
            
            if self.vlm_att_bert_proj is not None:
                self.vlm_att_bert_proj = self.vlm_att_bert_proj.to(device=device_type, dtype=weight_type)

    def forward(self, image_features, prompts=None, image_counts=None, long_video=False):        
        img_feat_lst = []
        # import pdb;pdb.set_trace()
        if image_counts is None:
            assert len(image_features) == len(prompts), f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
        else:
            assert len(prompts) == len(image_counts), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)    

        total_count = 0
        # calculate each image feat according to the prompt
        # import pdb;pdb.set_trace()
        for _idx in range(len(prompts)):
            assert isinstance(prompts[_idx], list), f"Prompt should be a list, but got {type(prompts[_idx])}"
            input_token = self.vlm_att_tokenlizer(
                prompts[_idx], 
                padding='longest', 
                truncation=True,
                max_length=256,
                return_tensors="pt"
                ).to(image_features.device)

            input_ids = input_token.input_ids
            attention_masks = input_token.attention_mask
            
            if image_counts is None:
                img_feat_prompt = image_features[_idx, None].expand(len(prompts[_idx]), -1, -1)
                img_att_prompt = image_atts[_idx, None].expand(len(prompts[_idx]), -1)
            else:
                # shape: [prompt_num*frame_num, image_shape, feat_dim]
                img_feat_prompt = image_features[total_count:total_count+image_counts[_idx]]
                img_feat_prompt = img_feat_prompt[None].expand(len(prompts[_idx]), -1, -1, -1).flatten(0,1)
                img_att_prompt = image_atts[total_count:total_count+image_counts[_idx]]
                img_att_prompt = img_att_prompt[None].expand(len(prompts[_idx]), -1, -1).flatten(0,1)
                input_ids = input_ids[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
                attention_masks = attention_masks[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
                total_count += image_counts[_idx]
            
            if "pretrain" in self.bert_type and self.vlm_att_bert_proj is not None:
                bert_feat = self.vlm_att_bert_proj(img_feat_prompt)
            else:
                bert_feat = img_feat_prompt.clone()

            # remove cls embedding
            if self.mm_vision_select_feature == 'patch':
                if img_feat_prompt.shape[1]%2 == 1:
                    img_feat_prompt = img_feat_prompt[:, 1:]

            if "qformer" in self.bert_type:
                query_tokens = self.vlm_att_query.expand(bert_feat.shape[0], -1, -1)
                query_atts = torch.cat([torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(bert_feat.device), 
                                        attention_masks],dim=1)
                
                if 'pretrain' in self.bert_type:
                    mm_img_in = self.vlm_att_ln(bert_feat)
                else:
                    mm_img_in = bert_feat
                
                if long_video:
                    outputs = []
                    block_size = 64
                    for L in range(0, len(input_ids), block_size):
                        R = L + block_size
                        mm_output = self.vlm_att_encoder.bert(
                            input_ids[L:R],
                            query_embeds=query_tokens[L:R],
                            attention_mask=query_atts[L:R],
                            encoder_hidden_states=mm_img_in[L:R],
                            encoder_attention_mask=img_att_prompt[L:R],
                            return_dict=True,
                        )
                        mm_output = mm_output.last_hidden_state[:,:query_tokens.shape[1]]
                        outputs.append(mm_output)
                    mm_output = torch.cat(outputs)
                    torch.cuda.empty_cache()
                else:
                    mm_output = self.vlm_att_encoder.bert(
                        input_ids,
                        query_embeds=query_tokens,
                        attention_mask=query_atts,
                        encoder_hidden_states=mm_img_in,
                        encoder_attention_mask=img_att_prompt,
                        return_dict=True,
                    )
                    mm_output = mm_output.last_hidden_state[:,:query_tokens.shape[1]]
                
            elif "raw" in self.bert_type:
                if self.mm_vision_select_feature == 'patch' and bert_feat.shape[1]%2 == 1:
                    bert_feat = bert_feat[:, 1:]
                    img_att_prompt = img_att_prompt[:, 1:]
                
                mm_output = self.vlm_att_encoder.bert(
                    input_ids,
                    attention_mask=attention_masks,
                    encoder_hidden_states=self.vlm_att_bert_proj(bert_feat),
                    encoder_attention_mask=img_att_prompt,
                    return_dict=True,
                )
                mm_output = mm_output.last_hidden_state
            else:
                raise ValueError(f'Unexpected bert type: {self.bert_type}')
            
            text_q = self.vlm_att_projector(mm_output)
            # shape: [prompt_num*frame_num, feat_dim]
            # ctx_embed,vis_embed = self.token_generation(text_q, img_feat_prompt, long_video=long_video)
            final_token = self.token_generation(text_q, img_feat_prompt, long_video=long_video)

            if image_counts is not None:
                # shape: [prompt_num, frame_num*image_shape, feat_dim]
                final_token = final_token.reshape(len(prompts[_idx]), image_counts[_idx], *final_token.shape[-2:])
                final_token = final_token.flatten(1,2)
            img_feat_lst.append(final_token)

        return img_feat_lst

    def init_bert(self, vision_width, cross_attention_freq=2, truncation_side="right"):
        # initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # initialize BERT
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        query_tokens = None
        
        if "qformer" in self.bert_type:
            mm_model = BertLMHeadModelQF.from_pretrained(
                "bert-base-uncased", config=encoder_config
            )
            query_tokens = nn.Parameter(
                torch.zeros(1, self.num_query, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        elif "raw" in self.bert_type:
            encoder_config.is_decoder = True
            mm_model = BertLMHeadModelRaw.from_pretrained(
                "bert-base-uncased", config=encoder_config
            )
        else:
            raise NotImplementedError("BERT type not implemented...")
        
        mm_model.resize_token_embeddings(len(tokenizer))
        mm_model.cls = None
        
        if "layer" in self.bert_type:
            layer_num = int(self.bert_type.split(':')[-1])
            mm_model.bert.encoder.layer = mm_model.bert.encoder.layer[:layer_num]
            print(f"Only use {layer_num} layers in BERT...")
        
        return tokenizer, mm_model, query_tokens


    def token_generation(self, text_q, vis_embed, long_video=False):
        ctx_embed = self.vlm_att_key_projector(vis_embed)
        # Key part 1: calculate context-related embedding
        ctx_embed = text_q @ ctx_embed.transpose(-1,-2) 
        ctx_embed = ctx_embed / (vis_embed.shape[-1] ** 0.5)
        if not long_video:
            ctx_embed = (ctx_embed.softmax(-1) @ vis_embed).mean(1)
        else:
            block_size = 64
            outputs = []
            ctx_score = ctx_embed.softmax(-1)    
            for L in range(0, len(ctx_score), block_size):
                R = L + block_size
                sub_embed = (ctx_score[L:R] @ vis_embed[L:R]).mean(1)
                outputs.append(sub_embed)
            ctx_embed = torch.cat(outputs)
            torch.cuda.empty_cache()
        ctx_embed = self.vlm_att_val_projector(ctx_embed[:,None])

        # Key part 2: calculate visual embedding
        if self.compress_type is not None:
            if 'grid' in self.compress_type:
                grid_size = int(self.compress_type.split('grid:')[-1])
                cur_shape = int(vis_embed.shape[1]**0.5)
                assert grid_size > 1, f'Grid size should be larger than 1, but got {grid_size}'
                vis_embed = vis_embed.reshape(vis_embed.shape[0], cur_shape, cur_shape, -1)
                grid_stride = cur_shape // grid_size
                vis_embed = F.avg_pool2d(vis_embed.permute(0, 3, 1, 2), 
                                         padding=0,
                                         kernel_size=grid_stride, 
                                         stride=grid_stride)
                
                vis_embed = vis_embed.permute(0, 2, 3, 1).flatten(1,2)
            elif 'mean' in self.compress_type:
                # import pdb;pdb.set_trace()
                vis_embed = vis_embed.mean(dim=1, keepdim=True)
        
        # import pdb ; pdb.set_trace()
        # concat token in shape (B, n+1, C)
        vis_embed = self.mm_projector(vis_embed)                
        final_token = torch.cat([ctx_embed, vis_embed], dim=1)
        return final_token

    @property
    def config(self):
        return {
            'mm_resampler_type': 'vlm_attention',
            'mm_vlmattention_bert_type': self.bert_type,
            'mm_vlmattention_num_query': self.num_query,
            'mm_vlmattention_compress_type': self.compress_type,
        }

