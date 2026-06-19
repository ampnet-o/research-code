from ampnet.modules.wavenet_conv import fast_1d_cconv

from typing import Callable
import time

import jax
import jax.numpy as jnp
from flax import nnx

# --------------------------------------------------
# benchmark fast cconv
# --------------------------------------------------
signal_len = 8192*2
batch_size = 64
n_kernels = 16
kernel_size = 8
n_runs = 100
dilation = 512

jit_fast_1d_cconv = jax.jit(
    fast_1d_cconv,
    static_argnums=(2,)
)

_nnx_conv = nnx.Conv(
    in_features=1,
    out_features=n_kernels,
    kernel_size=kernel_size,
    padding="CAUSAL",
    kernel_dilation=dilation,
    rngs=nnx.Rngs(0)
)

@jax.jit
def baseline_conv(x, k, dilation):
    return _nnx_conv(x)

def bench_conv(conv_impl: Callable):
    x = jnp.ones((batch_size, signal_len, 1))
    k = jnp.ones((n_kernels, kernel_size))

    # warmup
    conv_impl(x, k, dilation)
    print("warmup complete")

    t_sum = 0
    for n in range(n_runs):
        t_start = time.time()
        conv_impl(x, k, dilation).block_until_ready()
        t_end = time.time()
        t_sum += t_end - t_start
        x += 0.1

    print(f"avg. runtime:", t_sum / n_runs)

bench_conv(jit_fast_1d_cconv)
bench_conv(baseline_conv)


