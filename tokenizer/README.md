# Tokenizer

Implements tokenization from scratch — no HuggingFace, no shortcuts.

| Script | Covers |
|--------|--------|
| `Tokenizer.py` | Character/word-level tokenizer: vocab building, encode, decode |
| `Byte_Pair_Encoding.py` | BPE tokenizer: merges, subword vocabulary, encode/decode |

The `the-verdict.txt` file is used as the training corpus for both tokenizers.

## Setup

From the **project root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch
```

## Run

```bash
source ../.venv/bin/activate   # from inside tokenizer/
python Tokenizer.py
python Byte_Pair_Encoding.py
```
