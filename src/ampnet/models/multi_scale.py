# Multi scale RNNs as per LPCNet

import jax
import jax.numpy as jnp
from flax import nnx

from ..F import apply_rnn_burn
from ..utils.signal import overlap_add, apply_window, pad_time, init_legendre

import math


class TwinRNN(nnx.Module):
    """
    Wrapper arch consisting of 2 nets
    - Slow frame-based features network
    - sample-level generator conditioned on clean + frame features

    why:
        - LPCNet/FarGAN works well
        - split dry signal into dry[prediction + excitation]
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        frame_net: nnx.Module,       # (B, T_frame, d_frame) -> (B, T, d_cond)
        sample_net: nnx.Module,      # (B, T, d_in)          -> (B, T, d_out)
        rngs: nnx.Rngs,
        d_frame: int=8,
        hop_size: int=4,
        window=jnp.hanning
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.d_frame = d_frame
        self.hop_size = hop_size

        self.frame_net = frame_net
        self.sample_net = sample_net

        self.window = window(self.d_frame)

    def __call__(self, inputs: jax.Array, burn_steps: int=0) -> jax.Array:
        """
        inputs: (B, T, 1)
        """

        B, T, d = inputs.shape
        assert d == 1

        frames = apply_window(self.window, inputs.reshape(B, T), hop_size=self.hop_size)
        frame_features = self.frame_net(frames, hop=self.hop_size)
        # placeholder
        return jnp.zeros((B, T, d))


