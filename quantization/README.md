# Quantization

Covers model quantization techniques in order of complexity — from basic int8 to quantization-aware training.

| Script | Covers |
|--------|--------|
| `01_quantization_from_scratch.py` | Manual int8 quantization: scale, zero-point, quantize/dequantize |
| `02_quantization_compare_minmax_percentile.py` | Min-max vs percentile calibration strategies |
| `03_post_training_quantization.py` | PTQ: quantize a trained model's weights without retraining |
| `04_quantization_aware_training.py` | QAT: simulate quantization during training for better accuracy |

## Setup

From the **project root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch
```

## Run

```bash
source ../.venv/bin/activate   # from inside quantization/
python 01_quantization_from_scratch.py
python 02_quantization_compare_minmax_percentile.py
python 03_post_training_quantization.py
python 04_quantization_aware_training.py
```

Run in order — later scripts assume understanding from earlier ones.
