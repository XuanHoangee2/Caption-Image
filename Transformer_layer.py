import torch.nn as nn
import torch
import math 
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self,query,key, value,attn_mask = None):
        N, S, E = query.shape
        N, T, E = value.shape
        output = torch.empty(N,S,E)
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        Q = torch.tensor(Q)
        K = torch.tensor(K)
        V = torch.tensor(V)
        Q = Q.view(N,S,self.n_head,self.head_dim)
        K = K.view(N,T,self.n_head,self.head_dim)
        V = V.view(N,T,self.n_head,self.head_dim)
        Q = Q.permute(0, 2, 1, 3)  # (N, H, S, D)
        K = K.permute(0, 2, 1, 3)  # (N, H, T, D)
        V = V.permute(0, 2, 1, 3)   
        K = K.transpose(-2, -1) 
        scale = self.head_dim ** 0.5
        scores = torch.matmul(Q,K)/scale
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, S, T)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (N, 1, S, T)

            scores = scores.masked_fill(~attn_mask, float('-inf'))
        A = torch.softmax(scores, dim= -1)
        A = self.attn_drop(A)
        Y = torch.matmul(A, V)
        Y = Y.permute(0,2,1,3).contiguous()
        Y = Y.view(N,S,E)
        output = self.proj(Y)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0

        pe = torch.zeros(1, max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):

        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):

        out = torch.empty_like(x)

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim,num_heads,dropout)
        self.cross_attn = MultiHeadAttention(input_dim,num_heads,dropout)
        self.ffn = FeedForwardNetwork(input_dim,dim_feedforward,dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.drop_self = nn.Dropout(dropout) 
        self.drop_cross = nn.Dropout(dropout) 
        self.drop_fnn = nn.Dropout(dropout) 

    def forward(self,tgt,memory,tgt_mask = None):
        shortcut = tgt
        tgt = self.self_attn(query= tgt,key = tgt, value= tgt, attn_mask = tgt_mask)
        tgt = self.drop_self(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_self(tgt)

        shortcut_cross = tgt
        tgt = self.cross_attn(query = tgt, key = memory, value = memory,attn_mask = None)
        tgt = self.drop_cross(tgt)
        tgt = tgt + shortcut_cross
        tgt = self.norm_cross(tgt)

        shortcut_ffn = tgt
        tgt = self.ffn(tgt)
        tgt = self.drop_fnn(tgt)
        tgt = tgt + shortcut_ffn
        tgt = self.norm_ffn(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output