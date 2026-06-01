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
    return clamp(np.round(params / scale + zero), 0, 2**bits - 1).astype(np.int32), scale, zero


def asymmetric_quantization_percentile(params, bits, percentile=99.99):
    alpha = np.percentile(params, percentile)
    beta  = np.percentile(params, 100 - percentile)
    scale = (alpha - beta) / (2**bits - 1)
    zero  = -1 * np.round(beta / scale)
    return clamp(np.round(params / scale + zero), 0, 2**bits - 1).astype(np.int32), scale, zero


def asymmetric_dequantize(params_q, scale, zero):
    return (params_q - zero) * scale


def quantization_error(params, params_deq):
    return np.mean((params - params_deq) ** 2)


if __name__ == "__main__":
    params = np.random.uniform(low=-50, high=150, size=10000)
    params[-1] = 1000  # outlier
    params = np.round(params, 2)

    q_minmax, scale_mm, zero_mm = asymmetric_quantization(params, 8)
    q_pct,    scale_p,  zero_p  = asymmetric_quantization_percentile(params, 8)

    deq_minmax = asymmetric_dequantize(q_minmax, scale_mm, zero_mm)
    deq_pct    = asymmetric_dequantize(q_pct,    scale_p,  zero_p)

    # Compare error excluding the outlier
    err_mm  = quantization_error(params[:-1], deq_minmax[:-1])
    err_pct = quantization_error(params[:-1], deq_pct[:-1])
    print(f"Error (min-max, excl. outlier):    {err_mm:.4f}")
    print(f"Error (percentile, excl. outlier): {err_pct:.4f}")
