import torch
import torch.nn as nn

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = inputs.unsqueeze(0)  # shape [1, 6, 3]

d_in           = 3
d_out          = 4   # must be divisible by num_heads
num_heads      = 2
context_length = inputs.shape[0]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)    # [b, num_tokens, d_out]
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Split d_out into (num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # shape now: [b, num_heads, num_tokens, head_dim]

        attn_scores = queries @ keys.transpose(2, 3)  # [b, num_heads, num_tokens, num_tokens]
        mask_bool   = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)  # [b, num_tokens, num_heads, head_dim]
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # concatenate heads
        return self.out_proj(context_vec)

torch.manual_seed(123)
mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=num_heads)
context_vecs = mha(batch)
print("Output shape:", context_vecs.shape)  # [1, 6, 4]
print(context_vecs)

torch.manual_seed(123)
batch_gpt = torch.rand(2, 1024, 768)  # 2 sequences, 1024 tokens, 768-dim embeddings
mha_gpt   = MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0.0, num_heads=12)
out       = mha_gpt(batch_gpt)
print("GPT-2 scale output shape:", out.shape)  # [2, 1024, 768]
