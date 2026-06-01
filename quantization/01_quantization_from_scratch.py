import numpy as np

np.random.seed(0)
np.set_printoptions(suppress=True)


def clamp(params_q, lower_bound, upper_bound):
    params_q[params_q < lower_bound] = lower_bound
    params_q[params_q > upper_bound] = upper_bound
    return params_q


def asymmetric_quantization(params, bits):
    alpha, beta = np.max(params), np.min(params)
    scale = (alpha - beta) / (2**bits - 1)
    zero = -1 * np.round(beta / scale)
    quantized = clamp(
        np.round(params / scale + zero), 0, 2**bits - 1
    ).astype(np.int32)
    return quantized, scale, zero


def asymmetric_dequantize(params_q, scale, zero):
    return (params_q - zero) * scale


def symmetric_quantization(params, bits):
    alpha = np.max(np.abs(params))
    scale = alpha / (2**(bits - 1) - 1)
    lower_bound, upper_bound = -(2**(bits - 1)), 2**(bits - 1) - 1
    quantized = clamp(np.round(params / scale), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale


def symmetric_dequantize(params_q, scale):
    return params_q * scale


def quantization_error(params, params_deq):
    return np.mean((params - params_deq) ** 2)


if __name__ == "__main__":
    params = np.random.uniform(low=-50, high=150, size=20)
    params[0] = params.max() + 1
    params[1] = params.min() - 1
    params[2] = 0
    params = np.round(params, 2)

    asym_q, asym_scale, asym_zero = asymmetric_quantization(params, 8)
    sym_q, sym_scale = symmetric_quantization(params, 8)

    params_deq_asym = asymmetric_dequantize(asym_q, asym_scale, asym_zero)
    params_deq_sym  = symmetric_dequantize(sym_q, sym_scale)

    print(f"Asymmetric error: {quantization_error(params, params_deq_asym):.4f}")
    print(f"Symmetric error:  {quantization_error(params, params_deq_sym):.4f}")
