# Transformer / LLM Architecture

Full GPT-2 style transformer built from scratch — embedding, attention, feed-forward, layer norm, and text generation all in one place.

| Script | Covers |
|--------|--------|
| `LLM Architecture.py` | Token + positional embeddings, multi-head causal attention, LayerNorm, GELU, FFN, TransformerBlock, GPTModel, greedy/top-k text generation |

## Setup

From the **project root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch tiktoken
```

`tiktoken` is required for the GPT-2 tokenizer used in this script.

## Run

```bash
source ../.venv/bin/activate   # from inside transformer/
python "LLM Architecture.py"
```
