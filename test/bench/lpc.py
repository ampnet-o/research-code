from ampnet.utils.signal import lpc_windowed

import time
import numpy as np


# --------------------------------------------------
# benchmark lpc
# --------------------------------------------------

# params
signal_len = 8192*2
t_sum = 0
n_runs = 100
window_len = 256
order = 8
hop = window_len // 2   # default

signal = np.ones((1, signal_len))
window_mask = np.hanning(window_len)

# warmup jax
coeffs = lpc_windowed(signal=signal, window=window_mask, order=order, hop_size=hop)
print(coeffs.shape)

for i in range(n_runs):
    t_start = time.time()
    lpc_windowed(signal=signal, window=window_mask, order=order, hop_size=hop)
    t_end = time.time()
    t_sum += t_end - t_start
    signal += 0.1

print(f"total signal len: {signal_len}, window size: {window_len}")
print(f"avg. runtime:", t_sum / n_runs)



