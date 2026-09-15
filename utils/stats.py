import numpy as np

def mean_std(values):
    values = np.array(values, dtype=np.float32)
    return float(np.mean(values)), float(np.std(values))

def percentage_improvement(baseline, method):
    """
    Computes percentage improvement over baseline.
    """
    baseline = np.array(baseline, dtype=np.float32)
    method = np.array(method, dtype=np.float32)

    base_mean = np.mean(baseline)
    method_mean = np.mean(method)

    if base_mean == 0:
        return 0.0

    return float((method_mean - base_mean) / abs(base_mean) * 100.0)
