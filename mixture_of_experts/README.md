# Mixture of Experts (MoE)

Implements Mixture of Experts from scratch — the architecture behind models like Mixtral and GPT-4.

| File | Description |
|------|-------------|
| `10_mixture_of_experts.ipynb` | Interactive notebook — run in Colab or Jupyter |
| `10_mixture_of_experts.py` | Plain Python version of the same notebook |
| `05_mixture_of_experts.html` | Visual walkthrough of the MoE architecture |

## Setup

From the **project root**:

```bash
source .venv/bin/activate
pip install torch
```

## Run

```bash
source ../.venv/bin/activate   # from inside mixture_of_experts/
python 10_mixture_of_experts.py
```

## Visual Walkthrough

Open `05_mixture_of_experts.html` in a browser for an animated visual explanation of how expert routing works.
