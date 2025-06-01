import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
class SelfTransformerBlock(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        x1, x2, embedding = x
        norm_x1 = self.norm1(x1)
        attn_x1, _ = self.attn(norm_x1, norm_x1, norm_x1, need_weights=False)
        x1 = x1 + attn_x1
        norm_x1_2 = self.norm2(x1)
        x1 = self.mlp(norm_x1_2) + x1
        norm_x2 = self.norm1(x2)
        attn_x2, _ = self.attn(norm_x2, norm_x2, norm_x2, need_weights=False)
        x2 = x2 + attn_x2
        norm_x2_2 = self.norm2(x2)
        x2 = self.mlp(norm_x2_2) + x2
        return (x1, x2, embedding)
    

class SelfTransformerCatBlock(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        x1, x2, embedding = x
        x1_cat = torch.cat([x1, embedding], dim=1)
        x2_cat = torch.cat([x2, embedding], dim=1)
        norm_x1 = self.norm1(x1)
        norm_x1_cat = self.norm1(x1_cat)
        attn_x1, _ = self.attn(norm_x1, norm_x1_cat, norm_x1_cat, need_weights=False)
        x1 = x1 + attn_x1
        norm_x1_2 = self.norm2(x1)
        x1 = self.mlp(norm_x1_2) + x1
        norm_x2 = self.norm1(x2)
        norm_x2_cat = self.norm1(x2_cat)
        attn_x2, _ = self.attn(norm_x2, norm_x2_cat, norm_x2_cat, need_weights=False)
        x2 = x2 + attn_x2
        norm_x2_2 = self.norm2(x2)
        x2 = self.mlp(norm_x2_2) + x2
        return (x1, x2, embedding)


class DownSampleBlock(nn.Module):
    def __init__(self, feat_dim=1024, out_scale=2):

        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(out_scale)

    def forward(self, x):
        x1, x2, embedding = x
        embedding_reshaped = embedding.view(
            embedding.shape[0],
            int(embedding.shape[1] ** 0.5),
            int(embedding.shape[1] ** 0.5),
            -1,
        ).permute(0, 3, 1, 2)
        # print(embedding_reshaped.shape)
        embedding_pooled = self.pool(embedding_reshaped)
        embedding_pooled = (
            embedding_pooled.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        )
        norm_x1 = self.norm1(x1)
        attn_x1, _ = self.attn(embedding_pooled, norm_x1, norm_x1, need_weights=False)
        norm_x1_2 = self.norm2(attn_x1)
        res_x1 = self.mlp(norm_x1_2) + attn_x1
        norm_x2 = self.norm1(x2)
        attn_x2, _ = self.attn(embedding_pooled, norm_x2, norm_x2, need_weights=False)
        norm_x2_2 = self.norm2(attn_x2)
        res_x2 = self.mlp(norm_x2_2) + attn_x2
        return (res_x1, res_x2, embedding)


class CrossTransformerBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        x1, x2, embedding = x
        norm_x1 = self.norm1(x1)
        norm_x2 = self.norm1(x2)
        attn_x1, _ = self.attn(norm_x2, norm_x1, norm_x1, need_weights=False)
        x1 = x1 + attn_x1
        norm_x1_2 = self.norm2(x1)
        x1 = self.mlp(norm_x1_2) + x1
        attn_x2, _ = self.attn(norm_x1, norm_x2, norm_x2, need_weights=False)
        x2 = x2 + attn_x2
        norm_x2_2 = self.norm2(x2)
        x2 = self.mlp(norm_x2_2) + x2
        return (x1, x2, embedding)
    
class CrossTransformerCatBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        x1, x2, embedding = x
        x1_cat = torch.cat([x1, embedding], dim=1)
        x2_cat = torch.cat([x2, embedding], dim=1)
        norm_x1_cat = self.norm1(x1_cat)
        norm_x2_cat = self.norm1(x2_cat)
        norm_x1 = self.norm1(x1)
        norm_x2 = self.norm1(x2)
        attn_x1, _ = self.attn(norm_x2, norm_x1_cat, norm_x1_cat, need_weights=False)
        x1 = x1 + attn_x1
        norm_x1_2 = self.norm2(x1)
        x1 = self.mlp(norm_x1_2) + x1
        attn_x2, _ = self.attn(norm_x1, norm_x2_cat, norm_x2_cat, need_weights=False)
        x2 = x2 + attn_x2
        norm_x2_2 = self.norm2(x2)
        x2 = self.mlp(norm_x2_2) + x2
        return (x1, x2, embedding)
    

class QFormerLayer(nn.Module):
    def __init__(self, feat_dim=1152, num_heads=8, query_num=25):
        super().__init__()

        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn1 = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

        self.query_token = nn.Parameter(torch.randn(1, query_num, feat_dim))

    def forward(self, x):
        x1, x2, embedding = x
        attn_embedding, _ = self.attn1(self.query_token.repeat(embedding.shape[0], 1, 1), embedding, embedding, need_weights=False)
        norm_x1 = self.norm1(x1)
        attn_x1, _ = self.attn2(attn_embedding, norm_x1, norm_x1, need_weights=False)
        norm_x1_2 = self.norm2(attn_x1)
        x1 = self.mlp(norm_x1_2) + attn_x1
        norm_x2 = self.norm1(x2)
        attn_x2, _ = self.attn2(attn_embedding, norm_x2, norm_x2, need_weights=False)
        norm_x2_2 = self.norm2(attn_x2)
        x2 = self.mlp(norm_x2_2) + attn_x2
        return (x1, x2, embedding)


class SpatialQFormerLayer(nn.Module):
    def __init__(self, feat_dim=1152, num_heads=8, L=3):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(L**2, 1, feat_dim))
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn1 = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        self.L = L

    def forward(self, x):
        x1, x2, embedding = x
        B, N2, D = embedding.shape
        L = int(N2 ** 0.5)
        embedding_reshaped = embedding.view(B, L, L, -1)
        if L%self.L != 0:
            pad_target_size = (L // self.L + 1) * self.L
            
            pad_size = pad_target_size - L
            extra_pad_size = pad_size % 2
            pad_size = pad_size // 2
            
            # print(pad_target_size, pad_size, extra_pad_size)
            embedding_reshaped = F.pad(embedding_reshaped.permute(0, 3, 1, 2), (pad_size, pad_size+extra_pad_size, pad_size, pad_size+extra_pad_size), mode='constant', value=0).permute(0, 2, 3, 1)
            B, L, L, D = embedding_reshaped.shape

        embedding_reshaped = embedding_reshaped.view(B, L//self.L, self.L, L//self.L, self.L, D).permute(0, 1, 3, 5, 2, 4).contiguous().reshape(B*self.L*self.L, L//self.L*L//self.L, D)

        attn_embedding, _ = self.attn1(self.query_token.repeat(B, 1, 1), embedding_reshaped, embedding_reshaped, need_weights=False)
        # attn_embedding = attn_embedding.reshape(B, self.L**2, -1)
        norm_x1 = self.norm1(x1)

        B, N2, D = norm_x1.shape
        L = int(N2 ** 0.5)
        norm_x1 = norm_x1.view(B, L, L, -1)
        if L%self.L != 0:
            pad_target_size = (L // self.L + 1) * self.L
            
            pad_size = pad_target_size - L
            extra_pad_size = pad_size % 2
            pad_size = pad_size // 2
            
            # print(pad_target_size, pad_size, extra_pad_size)
            norm_x1 = F.pad(norm_x1.permute(0, 3, 1, 2), (pad_size, pad_size+extra_pad_size, pad_size, pad_size+extra_pad_size), mode='constant', value=0).permute(0, 2, 3, 1)
            B, L, L, D = norm_x1.shape

        norm_x1 = norm_x1.view(B, L//self.L, self.L, L//self.L, self.L, D).permute(0, 1, 3, 5, 2, 4).contiguous().reshape(B*self.L*self.L, L//self.L*L//self.L, D)
        attn_x1, _ = self.attn2(attn_embedding, norm_x1, norm_x1, need_weights=False)
        attn_x1 = attn_x1.reshape(B, self.L**2, -1)
        norm_x1_2 = self.norm2(attn_x1)
        x1 = self.mlp(norm_x1_2) + attn_x1

        norm_x2 = self.norm1(x2)
        B, N2, D = norm_x2.shape
        L = int(N2 ** 0.5)
        norm_x2 = norm_x2.view(B, L, L, -1)
        if L%self.L != 0:
            pad_target_size = (L // self.L + 1) * self.L

            pad_size = pad_target_size - L
            extra_pad_size = pad_size % 2
            pad_size = pad_size // 2
            
            # print(pad_target_size, pad_size, extra_pad_size)
            norm_x2 = F.pad(norm_x2.permute(0, 3, 1, 2), (pad_size, pad_size+extra_pad_size, pad_size, pad_size+extra_pad_size), mode='constant', value=0).permute(0, 2, 3, 1)
            B, L, L, D = norm_x2.shape

        norm_x2 = norm_x2.view(B, L//self.L, self.L, L//self.L, self.L, D).permute(0, 1, 3, 5, 2, 4).contiguous().reshape(B*self.L*self.L, L//self.L*L//self.L, D)

        attn_x2, _ = self.attn2(attn_embedding, norm_x2, norm_x2, need_weights=False)
        attn_x2 = attn_x2.reshape(B, self.L**2, -1)
        norm_x2_2 = self.norm2(attn_x2)
        x2 = self.mlp(norm_x2_2) + attn_x2

        return (x1, x2, embedding)
    
class DirectQformerBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        x1, x2, embedding = x
        norm_x1 = self.norm1(x1)
        attn_x1, _ = self.attn(embedding, norm_x1, norm_x1, need_weights=False)
        norm_x1_2 = self.norm2(attn_x1)
        x1 = self.mlp(norm_x1_2) + attn_x1
        norm_x2 = self.norm1(x2)
        attn_x2, _ = self.attn(embedding, norm_x2, norm_x2, need_weights=False)
        norm_x2_2 = self.norm2(attn_x2)
        x2 = self.mlp(norm_x2_2) + attn_x2
        return (x1, x2, embedding)
    
class QFormerCatLayer(nn.Module):
    def __init__(self, feat_dim=1152, num_heads=8, query_num=25):
        super().__init__()

        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn1 = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

        self.query_token = nn.Parameter(torch.randn(1, query_num, feat_dim))

    def forward(self, x):
        x1, x2, embedding = x
        x1 = torch.cat([x1, embedding], dim=1)
        x2 = torch.cat([x2, embedding], dim=1)
        attn_embedding, _ = self.attn1(self.query_token.repeat(embedding.shape[0], 1, 1), embedding, embedding, need_weights=False)
        norm_x1 = self.norm1(x1)
        attn_x1, _ = self.attn2(attn_embedding, norm_x1, norm_x1, need_weights=False)
        norm_x1_2 = self.norm2(attn_x1)
        x1 = self.mlp(norm_x1_2) + attn_x1
        norm_x2 = self.norm1(x2)
        attn_x2, _ = self.attn2(attn_embedding, norm_x2, norm_x2, need_weights=False)
        norm_x2_2 = self.norm2(attn_x2)
        x2 = self.mlp(norm_x2_2) + attn_x2
        return (x1, x2, embedding)
    
class DirectQformerCatBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        x1, x2, embedding = x
        x1 = torch.cat([x1, embedding], dim=1)
        x2 = torch.cat([x2, embedding], dim=1)
        norm_x1 = self.norm1(x1)
        attn_x1, _ = self.attn(embedding, norm_x1, norm_x1, need_weights=False)
        norm_x1_2 = self.norm2(attn_x1)
        x1 = self.mlp(norm_x1_2) + attn_x1
        norm_x2 = self.norm1(x2)
        attn_x2, _ = self.attn(embedding, norm_x2, norm_x2, need_weights=False)
        norm_x2_2 = self.norm2(attn_x2)
        x2 = self.mlp(norm_x2_2) + attn_x2
        return (x1, x2, embedding)
    
class DirectOutputBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        
    def forward(self, x):
        x2, embedding = x
        xx = x2 + embedding
        return x2, xx


class OutputBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        x1, x2, embedding = x
        # x2 = self.norm1(x2)
        xx = torch.cat([embedding, x2], dim=1)
        norm_xx = self.norm1(xx)
        attn_xx, _ = self.attn(embedding, norm_xx, norm_xx, need_weights=False)
        xx = attn_xx + embedding
        # attn_xx = attn_xx + embedding
        # norm_xx_2 = self.norm2(attn_xx)
        # xx = self.mlp(norm_xx_2) + attn_xx
        return x1, x2, xx


class DeltaSelfTransformerBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        delta, embedding = x
        norm_delta = self.norm1(delta)
        attn_delta, _ = self.attn(norm_delta, norm_delta, norm_delta, need_weights=False)
        attn_delta = attn_delta + delta
        norm_delta_2 = self.norm2(attn_delta)
        delta = self.mlp(norm_delta_2) + attn_delta
        return delta, embedding


class DeltaUpSampleCrossTransformerBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        delta, embedding = x
        norm_delta = self.norm1(delta)
        attn_delta, _ = self.attn(embedding, norm_delta, norm_delta, need_weights=False)
        norm_delta_2 = self.norm2(attn_delta)
        delta = self.mlp(norm_delta_2) + attn_delta
        return delta, embedding


class DeltaCrossTransformerBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        delta, embedding = x
        norm_delta = self.norm1(delta)
        attn_delta, _ = self.attn(embedding, norm_delta, norm_delta, need_weights=False)
        attn_delta_plus = attn_delta + delta
        norm_delta_2 = self.norm2(attn_delta_plus)
        delta = self.mlp(norm_delta_2) + attn_delta_plus
        return delta, embedding

class DeltaCrossCatBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        delta, embedding = x
        x1 = torch.cat([delta, embedding], dim=1)
        norm_x1 = self.norm1(x1)
        attn_x1, _ = self.attn(embedding, norm_x1, norm_x1, need_weights=False)
        norm_x1_2 = self.norm2(attn_x1)
        x1 = self.mlp(norm_x1_2) + attn_x1
        return x1, embedding
    
class DeltaSpatialQFormerLayer(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8, L=8):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(L**2, 1, feat_dim))
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn1 = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        self.L = L

    def forward(self, x):
        delta, embedding = x
        B, N2, D = embedding.shape
        L = int(N2 ** 0.5)
        embedding_reshaped = embedding.view(B, L, L, -1)
        if L%self.L != 0:
            pad_target_size = (L // self.L + 1) * self.L
            
            pad_size = pad_target_size - L
            extra_pad_size = pad_size % 2
            pad_size = pad_size // 2
            
            # print(pad_target_size, pad_size, extra_pad_size)
            embedding_reshaped = F.pad(embedding_reshaped.permute(0, 3, 1, 2), (pad_size, pad_size+extra_pad_size, pad_size, pad_size+extra_pad_size), mode='constant', value=0).permute(0, 2, 3, 1)
            B, L, L, D = embedding_reshaped.shape

        embedding_reshaped = embedding_reshaped.view(B, L//self.L, self.L, L//self.L, self.L, D).permute(0, 1, 3, 5, 2, 4).contiguous().reshape(B*self.L*self.L, L//self.L*L//self.L, D)
        # print(embedding_reshaped.shape)
        attn_embedding, _ = self.attn1(self.query_token.repeat(B, 1, 1), embedding_reshaped, embedding_reshaped, need_weights=False)
        # attn_embedding = attn_embedding.reshape(B, self.L**2, -1)
        norm_x1 = self.norm1(delta)

        B, N2, D = norm_x1.shape
        L = int(N2 ** 0.5)
        norm_x1 = norm_x1.view(B, L, L, -1)
        if L%self.L != 0:
            pad_target_size = (L // self.L + 1) * self.L
            
            pad_size = pad_target_size - L
            extra_pad_size = pad_size % 2
            pad_size = pad_size // 2
            
            # print(pad_target_size, pad_size, extra_pad_size)
            norm_x1 = F.pad(norm_x1.permute(0, 3, 1, 2), (pad_size, pad_size+extra_pad_size, pad_size, pad_size+extra_pad_size), mode='constant', value=0).permute(0, 2, 3, 1)
            B, L, L, D = norm_x1.shape

        norm_x1 = norm_x1.view(B, L//self.L, self.L, L//self.L, self.L, D).permute(0, 1, 3, 5, 2, 4).contiguous().reshape(B*self.L*self.L, L//self.L*L//self.L, D)
        attn_x1, _ = self.attn2(attn_embedding, norm_x1, norm_x1, need_weights=False)
        attn_x1 = attn_x1.reshape(B, self.L**2, -1)
        norm_x1_2 = self.norm2(attn_x1)
        x1 = self.mlp(norm_x1_2) + attn_x1

        return x1, embedding
    
class DeltaOutputBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )
    
    def forward(self, x):
        delta, embedding = x
        norm_delta = self.norm1(delta)
        attn_delta, _ = self.attn(embedding, norm_delta, norm_delta, need_weights=False)
        norm_delta_2 = self.norm2(attn_delta)
        pred_delta = self.mlp(norm_delta_2) + attn_delta + embedding
        return delta, pred_delta
    

class DeltaEmbeddingTransformerBlock(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        delta, embedding = x
        norm_embedding = self.norm1(embedding)
        attn_embedding, _ = self.attn(norm_embedding, norm_embedding, norm_embedding, need_weights=False)
        attn_embedding = attn_embedding + embedding
        norm_embedding_2 = self.norm2(attn_embedding)
        embedding = self.mlp(norm_embedding_2) + attn_embedding
        return delta, embedding

class DeltaPredictor(nn.Module):
    def __init__(self, delta_dim=9, feat_dim=1152):
        super().__init__()
        self.upsample = nn.Linear(delta_dim, 169)
        self.pos_embedding = nn.Parameter(torch.randn((384 // 16) ** 2, feat_dim))
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=16, p2=16),
            nn.Linear(768, feat_dim),
        )
        self.blocks = nn.Sequential(
            DeltaSelfTransformerBlock(feat_dim=feat_dim),
            DeltaEmbeddingTransformerBlock(feat_dim=feat_dim),
            DeltaUpSampleCrossTransformerBlock(feat_dim=feat_dim),
            DeltaSelfTransformerBlock(feat_dim=feat_dim),
            DeltaEmbeddingTransformerBlock(feat_dim=feat_dim),
            DeltaCrossTransformerBlock(feat_dim=feat_dim),
            DeltaSelfTransformerBlock(feat_dim=feat_dim),
            DeltaEmbeddingTransformerBlock(feat_dim=feat_dim),
            DeltaCrossTransformerBlock(feat_dim=feat_dim),
            DeltaSelfTransformerBlock(feat_dim=feat_dim),
            DeltaEmbeddingTransformerBlock(feat_dim=feat_dim),
            DeltaCrossTransformerBlock(feat_dim=feat_dim),
            # DeltaSelfTransformerBlock(feat_dim=feat_dim),
            # DeltaCrossTransformerBlock(feat_dim=feat_dim),
            # DeltaSelfTransformerBlock(feat_dim=feat_dim),
            # DeltaCrossTransformerBlock(feat_dim=feat_dim),
            # DeltaSelfTransformerBlock(feat_dim=feat_dim),
            # DeltaCrossTransformerBlock(feat_dim=feat_dim),
            # DeltaSelfTransformerBlock(feat_dim=feat_dim),
            # DeltaCrossTransformerBlock(feat_dim=feat_dim),
            DeltaSelfTransformerBlock(feat_dim=feat_dim),
            # DirectOutputBlock(feat_dim=feat_dim),
        )

        # self.blocks = nn.Sequential(
        #     # SelfTransformerBlock(feat_dim),
        #     # CrossTransformerBlock(feat_dim),
        #     # SelfTransformerBlock(feat_dim),
        #     # CrossTransformerBlock(feat_dim),
        #     # SpatialQFormerLayer(feat_dim, L=8),
        #     # QFormerLayer(feat_dim, query_num=64),
        #     # SelfTransformerBlock(feat_dim),
        #     # CrossTransformerBlock(feat_dim),
        #     SelfTransformerCatBlock(feat_dim),
        #     CrossTransformerCatBlock(feat_dim),
        #     # SpatialQFormerLayer(feat_dim, L=5),
        #     # DirectQformerBlock(feat_dim),
        #     # DirectQformerCatBlock(feat_dim),
        #     # QFormerCatLayer(feat_dim, query_num=25),
        #     DirectQformerBlock(feat_dim),
        #     # SelfTransformerBlock(feat_dim),
        #     # CrossTransformerBlock(feat_dim),
        #     SelfTransformerCatBlock(feat_dim),
        #     CrossTransformerCatBlock(feat_dim),
        #     OutputBlock(feat_dim),
        # )

    def forward(self, figure_origin, figure_new, embedding_origin=None):
        # embedding_origin: [T, 169, 1152]
        figure_delta = figure_new - figure_origin

        figure_delta = self.to_patch_embedding(figure_delta) + self.pos_embedding

        out, pred = self.blocks((figure_delta, embedding_origin))

        pred = out + embedding_origin

        return out, pred

        # figure_origin = self.to_patch_embedding(figure_origin) + self.pos_embedding
        # figure_new = (
        #     self.to_patch_embedding(figure_new) + self.pos_embedding
        # )  # [T, 384/16**2, 1152]

        # figure_origin, figure_new, new_frame = self.blocks(
        #     (figure_origin, figure_new, embedding_origin)
        # )

        # return figure_new, new_frame