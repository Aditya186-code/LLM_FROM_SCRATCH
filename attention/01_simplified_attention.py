import torch


inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your
     [0.55, 0.87, 0.66],  # journey
     [0.57, 0.85, 0.64],  # starts
     [0.22, 0.58, 0.33],  # with
     [0.77, 0.25, 0.10],  # one
     [0.05, 0.80, 0.55]]  # step
)


def simplified_attention(inputs, query_idx=1):
    query = inputs[query_idx]
    attn_scores = torch.tensor([torch.dot(x, query) for x in inputs])
    attn_weights = torch.softmax(attn_scores, dim=0)
    context_vec = attn_weights @ inputs
    return attn_weights, context_vec


def attention_all_tokens(inputs):
    attn_scores = inputs @ inputs.T
    attn_weights = torch.softmax(attn_scores, dim=-1)
    return attn_weights @ inputs


if __name__ == "__main__":
    weights, ctx = simplified_attention(inputs, query_idx=1)
    print("Attention weights (query=token 1):", weights)
    print("Context vector:", ctx)

    context_vecs = attention_all_tokens(inputs)
    print("\nContext vectors (all tokens):\n", context_vecs)
