# Attention Mechanisms

Builds up attention from scratch in four steps — each script is self-contained and runnable.

| Script | Covers |
|--------|--------|
| `01_simplified_attention.py` | Naive dot-product attention with no learned weights |
| `02_self_attention_qkv.py` | Self-attention with Query, Key, Value projections |
| `03_causal_attention.py` | Causal (masked) attention — tokens only attend to past positions |
| `04_multi_head_attention.py` | Multi-head attention as used in transformers |

## Setup

From the **project root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch
```

## Run

```bash
source ../.venv/bin/activate   # from inside attention/
python 01_simplified_attention.py
python 02_self_attention_qkv.py
python 03_causal_attention.py
python 04_multi_head_attention.py
```

Run them in order — each builds on concepts from the previous.
