from ampnet.utils.signal import lpc_windowed

import jax.numpy as jnp
import numpy as np

def test_lpc_windowed(signal_len=8192):
    signal = np.ones((1, signal_len))
    window_mask = np.hanning(32)
    coeffs = lpc_windowed(signal=signal, window=window_mask, order=8, hop_size=4)
    return coeffs


