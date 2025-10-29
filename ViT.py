import torch
import torch.nn as nn
import numpy as np
from Transformer_layer import MultiHeadAttention,FeedForwardNetwork

class PatchEmbedding(nn.Module):
    def __init__(self,img_size,patch_size,in_channel = 3,embedding_dim = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patch = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channel

        self.proj = nn.Linear(self.patch_dim,embedding_dim)

    def forward(self,x):
        N,C,H,W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Expected img size ({self.img_size}, {self.img_size}), but got ({H},{W})"
        
        out = torch.zeros(N,self.embedding_dim)
        x = x.reshape(N, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(N, self.num_patch, self.patch_dim)
        out = self.proj(x)
        return out
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward = 2048, dropout = 0.1):
        super().__init__()
        self.self_atten = MultiHeadAttention(input_dim,num_heads,dropout)
        self.ffn = FeedForwardNetwork(input_dim,num_heads,dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
    
    def forward(self,src, src_mask = None):
        attn_out = self.self_atten(src, src, src,attn_mask = src_mask)  
        src = src + self.dropout_self(attn_out)   
        src = self.norm_self(src)

        ffn_out = self.ffn(src)
        src = src + self.dropout_ffn(ffn_out)    
        src = self.norm_ffn(src)

        return src

torch.manual_seed(231)
np.random.seed(231)

N = 2
HW = 16
PS = 8
D = 8

patch_embedding = PatchEmbedding(img_size=HW,patch_size=PS,embedding_dim=D)

x = torch.randn(N, 3, HW, HW)
output = patch_embedding(x)
print(output)

N, T, TM, D = 1, 4, 5, 12

encoder_layer = TransformerEncoderLayer(D, 2, 4*D)
x = torch.randn(N, T, D)
x_mask = torch.randn(T, T) < 0.5

output = encoder_layer(x, x_mask)
print(output)