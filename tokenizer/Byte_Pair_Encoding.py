import importlib

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids  = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i: i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1: i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256, stride=128,
                      shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset   = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    print("tiktoken version:", importlib.metadata.version("tiktoken"))

    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("Encoded:", ids)
    print("Decoded:", tokenizer.decode(ids))

    # BPE handles unknown words by splitting into subword units
    print("Unknown word:", tokenizer.decode(tokenizer.encode("Akwirw ier")))

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    inputs, targets = next(iter(dataloader))
    print("\nInputs shape:", inputs.shape)
    print("Targets shape:", targets.shape)

    # Token + positional embeddings
    vocab_size    = 50257
    output_dim    = 256
    max_length    = 4
    token_emb     = torch.nn.Embedding(vocab_size, output_dim)
    pos_emb       = torch.nn.Embedding(max_length, output_dim)
    input_embeds  = token_emb(inputs) + pos_emb(torch.arange(max_length))
    print("Input embeddings shape:", input_embeds.shape)
