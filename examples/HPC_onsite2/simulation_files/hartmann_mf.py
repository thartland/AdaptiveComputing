"""Low-fidelity variants of the 6D Hartmann function for multi-fidelity benchmarks.

The high-fidelity function is ``hartmann.hf_hartmann`` (global minimum -3.32237).

lf_close
    Hartmann with the first weight lowered from 1.0 to 0.9, the low-fidelity
    form used in standard multi-fidelity Hartmann benchmarks (e.g. BoTorch's
    AugmentedHartmann at fidelity 0).  Very close to HF, same optimum.
lf_forrester
    The 6D analogue of the Forrester (2007) low fidelity that Tucker's 1D
    example already uses (0.5 f + 10 (x - 0.5) - 5): a scaled HF plus a linear
    trend and an offset.  Clearly biased, but the optimum stays in the HF basin.
"""
import numpy as np

try:
    from .hartmann import hf_hartmann
except ImportError:
    from hartmann import hf_hartmann

_A = np.array([[10., 3.0, 17., 3.5, 1.7, 8.0],
               [0.05, 10., 17., 0.1, 8.0, 14.],
               [3.0, 3.5, 1.7, 10., 17., 8.0],
               [17., 8.0, 0.05, 10., 0.1, 14.]])
_P = 1.e-4 * np.array([[1312., 1696., 5569., 124., 8283., 5886.],
                       [2329., 4135., 8307., 3736., 1004., 9991.],
                       [2348., 1451., 3522., 2883., 3047., 6650.],
                       [4047., 8828., 8732., 5743., 1091., 381.0]])


def lf_close(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    alpha = np.array([0.9, 1.2, 3.0, 3.2])
    y = -np.sum(alpha * np.exp(-np.sum(_A * (x - _P) ** 2, axis=1)))
    return np.array([[y]])


def lf_forrester(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = 0.5 * hf_hartmann(x).item() + 0.5 * (np.mean(x) - 0.5) - 0.5
    return np.array([[y]])
