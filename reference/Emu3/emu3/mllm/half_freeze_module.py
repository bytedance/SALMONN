# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

import torch
import torch.nn as nn

# [Tang] add learnable embedding
class HalfFreezeEmbedding(nn.Module):
    def __init__(self, original_embedding, additional_embedding_num):
        super(HalfFreezeEmbedding, self).__init__()

        self.original_embedding = nn.Embedding.from_pretrained(original_embedding, freeze=True)
        # self.original_embedding.to(torch.float32)
        self.orgemb_num = self.original_embedding.weight.size(0)
        self.embedding_dim = original_embedding.size(1)
        self.additional_embedding = nn.Parameter(torch.randn(additional_embedding_num, original_embedding.size(1), dtype=torch.float32), requires_grad=True)

    def forward(self, indices):
        original_mask = indices < self.original_embedding.num_embeddings
        original_indices = indices[original_mask]
        # original_result = self.original_embedding(original_indices).to(torch.float32)
        original_result = self.original_embedding(original_indices)

        additional_mask = ~original_mask
        additional_indices = indices[additional_mask] - self.orgemb_num
        # additional_result = self.additional_embedding[additional_indices]
        additional_result = self.additional_embedding[additional_indices].to(original_result.dtype)

        # result = original_result.new_zeros(*indices.shape, self.embedding_dim)
        # result[original_mask] = original_result
        # result[additional_mask] = additional_result

        result = torch.zeros(*indices.shape, self.embedding_dim, dtype=original_result.dtype).to(original_mask.device)
        result[original_mask] = original_result
        result[additional_mask] = additional_result

        return result

class HalfFreezeLinear(nn.Module):
    def __init__(self, original_linear, new_dim):
        super(HalfFreezeLinear, self).__init__()
        self.original_linear = original_linear
        self.new_linear = nn.Linear(original_linear.in_features, new_dim, bias=False)
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        x1 = self.original_linear(x)
        x2 = self.new_linear(x)
        y = torch.cat([x1, x2], dim=-1)
        return y

if __name__ == "__main__":
    original_embedding_dim = 10
    additional_embedding_dim = 5
    original_embedding_matrix = torch.randn(10, original_embedding_dim)  

    custom_embedding = HalfFreezeEmbedding(original_embedding_matrix, additional_embedding_dim)

    indices = torch.tensor([[5, 12], [13, 1]]) 
    output = custom_embedding(indices)
    print(output)