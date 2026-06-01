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

# Add batch dimension: shape [1, 6, 3]
batch = inputs.unsqueeze(0)

d_in = inputs.shape[1]
d_out = 2
context_length = inputs.shape[0]

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print("Mask (1 = blocked future position):\n", mask)

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out    = d_out
        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout  = nn.Dropout(dropout)
        # register_buffer ensures mask moves to the same device as the model (CPU/GPU)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        # mask_fill_ is in-place; slicing [:num_tokens] handles sequences shorter than context_length
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values

torch.manual_seed(123)
ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
context_vecs = ca(batch)
print("Output shape:", context_vecs.shape)  # [1, 6, 2]
print(context_vecs)
