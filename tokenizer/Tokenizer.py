import re


def build_vocab(text):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [t.strip() for t in tokens if t.strip()]
    return {token: i for i, token in enumerate(sorted(set(tokens)))}


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        return [self.str_to_int[t] for t in tokens]

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.?!()\'"\]])', r'\1', text)


class SimpleTokenizerV2:
    """Like V1 but handles out-of-vocabulary words with <|unk|>."""

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        tokens = [t if t in self.str_to_int else "<|unk|>" for t in tokens]
        return [self.str_to_int[t] for t in tokens]

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?!()\'"\]])', r'\1', text)


if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab = build_vocab(raw_text)
    print(f"Vocab size: {len(vocab)}")

    tokenizer_v1 = SimpleTokenizerV1(vocab)
    sample = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    ids = tokenizer_v1.encode(sample)
    print("Encoded:", ids[:10], "...")
    print("Decoded:", tokenizer_v1.decode(ids))

    # V2 with special tokens
    all_tokens = sorted(set(re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)))
    all_tokens = [t.strip() for t in all_tokens if t.strip()]
    all_tokens += ["<|endoftext|>", "<|unk|>"]
    vocab_v2 = {token: i for i, token in enumerate(all_tokens)}

    tokenizer_v2 = SimpleTokenizerV2(vocab_v2)
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
    print("\nV2 encoded:", tokenizer_v2.encode(text))
    print("V2 decoded:", tokenizer_v2.decode(tokenizer_v2.encode(text)))
