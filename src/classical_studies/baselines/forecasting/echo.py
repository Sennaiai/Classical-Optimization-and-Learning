import numpy as np


def finite_windows(windows, features=7):
    x = np.asarray(windows, dtype=float)
    if (
        x.ndim != 3
        or x.shape[0] == 0
        or not 1 <= x.shape[1] <= 48
        or x.shape[2] != features
    ):
        raise ValueError("Expected nonempty [samples, time<=48, seven grid features]")
    if not np.isfinite(x).all():
        raise ValueError("Reservoir input must be finite")
    return x


def echo_features(windows, seed, width=33):
    """Classical nonlinear fixed-weight window reservoir, matched spin dimension."""
    x = finite_windows(windows)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.5 / np.sqrt(7), (7, width))
    recurrent = rng.normal(0, 1 / np.sqrt(width), (width, width))
    recurrent *= 0.8 / max(abs(np.linalg.eigvals(recurrent)))
    h = np.zeros((len(x), width))
    for t in range(x.shape[1]):
        h = np.tanh(x[:, t] @ w + h @ recurrent)
    return h
