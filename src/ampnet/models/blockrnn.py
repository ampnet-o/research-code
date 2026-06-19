import jax
import jax.numpy as jnp
from flax import nnx

from ..F import apply_rnn_burn
from ..utils.signal import overlap_add, apply_window, pad_time, init_legendre

import math

class BlockRNN(nnx.Module):
    """
    naive audio block based RNN with linear readout

    why:
        - single sample RNN would be too slow
    """
    def __init__(
        self,
        rngs,
        d_model: int=128,
        d_block: int=8,
    ):
        self.d_block = d_block
        self.rnn = nnx.RNN(
            nnx.SimpleCell(
                in_features=d_block,
                hidden_features=d_model,
                rngs=rngs
            ),
            return_carry=False,
        )
        self.out_conv = nnx.Conv(in_features=1, out_features=1, kernel_size=(8,), rngs=rngs)

        self.dense = nnx.Linear(in_features=d_model, out_features=d_block, rngs=rngs)
        self.d_model = d_model

    def __call__(self, inputs: jax.Array, burn_steps: int=0):
        """
        inputs: (B, T, 1) or (B, T_block, d_block)
        """

        B, T, d = inputs.shape
        if d == 1:
            inputs = pad_time(inputs, T % self.d_block).reshape(B, -(T//-self.d_block), self.d_block)

        if burn_steps > 0:
            burn_steps = burn_steps // self.d_block
            x = apply_rnn_burn(self.rnn, inputs, burn_steps)[:, burn_steps:]
            inputs = inputs[:, burn_steps:].reshape(B, -1, 1)
        else:
            x = self.rnn(inputs)
            assert isinstance(x, jax.Array)
            inputs = inputs.reshape(B, -1, 1)

        x = self.dense(x.reshape(-1, self.d_model))
        x = x.reshape(B, -1, 1)
        x = self.out_conv(x)

        return x + inputs

# --------------------------------------------------

class StackBlockRNN(nnx.Module):
    """
    naive audio block based RNN + a smaller RNN-based readout

    why:
        - another approach to the smoothness problem with blocks
    """
    def __init__(
        self,
        rngs,
        d_model: int=128,
        d_block: int=8,
        d_readout: int=4,
    ):
        self.d_block = d_block
        self.d_readout = d_readout
        self.rnn = nnx.RNN(
            nnx.SimpleCell(
                in_features=d_block,
                hidden_features=d_model,
                rngs=rngs
            ),
            return_carry=False,
        )

        self.readout_rnn = nnx.RNN(
            nnx.SimpleCell(
                in_features=1,
                hidden_features=d_readout,   # TODO: parameterize?
                rngs=rngs
            ),
            return_carry=False,
        )
        self.dense = nnx.Linear(in_features=d_model, out_features=d_block, rngs=rngs)
        self.out_proj = nnx.Linear(in_features=d_readout, out_features=1, rngs=rngs)
        self.act = nnx.gelu
        self.d_model = d_model

    def __call__(self, inputs: jax.Array, burn_steps: int=0):
        """
        inputs: (B, T, 1) or (B, T_block, d_block)
        """

        B, T, d = inputs.shape
        if d == 1:
            inputs = pad_time(inputs, T % self.d_block).reshape(B, -(T//-self.d_block), self.d_block)

        if burn_steps > 0:
            bburn_steps = burn_steps // self.d_block
            x = apply_rnn_burn(self.rnn, inputs, burn_steps)[:, bburn_steps:]
            inputs = inputs[:, bburn_steps:].reshape(B, -1, 1)
        else:
            x = self.rnn(inputs)
            assert isinstance(x, jax.Array)
            inputs = inputs.reshape(B, T, 1)

        x = self.dense(x.reshape(-1, self.d_model))
        x = self.act(x)
        x = x.reshape(B, -1, 1)

        # BACK: burnout on readout rnn
        x = self.readout_rnn(x)
        x = self.out_proj(x)

        return x + inputs


class PolyWindowRNN(nnx.Module):
    """
    polynomial bf represented blocks + hann windowing

    why:
        - Linear readout blocks seem to be too jagged
    """
    def __init__(
        self,
        rngs,
        d_model: int=128,
        d_block: int=8,
        hop: int=4,
        poly_deg: int=4
    ):
        self.d_block = d_block
        self.rnn = nnx.RNN(
            nnx.SimpleCell(
                in_features=d_block,
                hidden_features=d_model,
                rngs=rngs
            ),
            return_carry=False,
        )
        self.window = jnp.hanning(d_block)
        self.hop = hop
        self.dense = nnx.Linear(in_features=d_model, out_features=d_block, rngs=rngs)
        self.d_model = d_model
        self.poly_readout = init_legendre(d_block, poly_deg)

    def __call__(self, inputs: jax.Array, burn_steps: int=0):
        """
        inputs: (B, T, 1)
        """

        B, T, d = inputs.shape
        assert d == 1

        x_in = apply_window(self.window, inputs.reshape(B, -1), self.hop)

        if burn_steps > 0:

            # TODO: block burn is imperfect (overestimate)
            block_burn = burn_steps // self.d_block
            x = apply_rnn_burn(self.rnn, x_in, block_burn)
            t_burn = burn_steps
        else:
            x = self.rnn(x_in)
            assert isinstance(x, jax.Array)
            t_burn = 0

        x = self.dense(x.reshape(-1, self.d_model))
        x = x*(self.window[None, ...])

        x = x.reshape(B, -1, self.d_block)
        x = overlap_add(x, self.hop)
        x = x.reshape(B, -1, 1)

        res = x[:, t_burn:, :] + inputs[:, t_burn:, :]
        return res


class WindowRNN(nnx.Module):
    """
    very basic windowing + linear block based RNN

    why:
        - single sample RNN would be too slow
        - another approach to boundary oscillations
    """
    def __init__(
        self,
        rngs,
        d_model: int=128,
        d_block: int=8,
        hop: int=4
    ):
        self.d_block = d_block
        self.rnn = nnx.RNN(
            nnx.SimpleCell(
                in_features=d_block,
                hidden_features=d_model,
                rngs=rngs
            ),
            return_carry=False,
        )
        self.window = jnp.hanning(d_block)
        self.hop = hop
        self.dense = nnx.Linear(in_features=d_model, out_features=d_block, rngs=rngs)
        self.d_model = d_model

    def __call__(self, inputs: jax.Array, burn_steps: int=0):
        """
        inputs: (B, T, 1)
        """

        B, T, d = inputs.shape
        assert d == 1

        x_in = apply_window(self.window, inputs.reshape(B, -1), self.hop)

        if burn_steps > 0:

            # TODO: block burn is imperfect (overestimate)
            block_burn = burn_steps // self.d_block
            x = apply_rnn_burn(self.rnn, x_in, block_burn)
            t_burn = burn_steps
        else:
            x = self.rnn(x_in)
            assert isinstance(x, jax.Array)
            t_burn = 0

        x = self.dense(x.reshape(-1, self.d_model))
        x = x*(self.window[None, ...])

        x = x.reshape(B, -1, self.d_block)
        x = overlap_add(x, self.hop)
        x = x.reshape(B, -1, 1)

        res = x[:, t_burn:, :] + inputs[:, t_burn:, :]
        return res


# NOT IMPLEMENTED
class SplineRNN(nnx.Module):
    """
    cubic spline + block based RNN

    why:
        - single sample RNN would be too slow
        - polynomial/linear blocks have boundary oscillations
    """
    def __init__(
        self,
        rngs,
        d_model: int=128,
        d_block: int=8,
    ):
        self.d_block = d_block
        self.rnn = nnx.RNN(
            nnx.SimpleCell(
                in_features=d_block,
                hidden_features=d_model,
                rngs=rngs
            ),
            return_carry=False,
        )
        self.out_conv = nnx.Conv(in_features=1, out_features=1, kernel_size=(8,), rngs=rngs)

        self.dense = nnx.Linear(in_features=d_model, out_features=d_block, rngs=rngs)
        self.d_model = d_model

    def __call__(self, inputs: jax.Array, burn_steps: int=0):
        """
        inputs: (B, T, 1) or (B, T_block, d_block)
        """

        B, T, d = inputs.shape
        if d == 1:
            inputs = pad_time(inputs, T % self.d_block).reshape(B, -(T//-self.d_block), self.d_block)

        if burn_steps > 0:
            burn_steps = burn_steps // self.d_block
            x = apply_rnn_burn(self.rnn, inputs, burn_steps)[:, burn_steps:]
            inputs = inputs[:, burn_steps:].reshape(B, -1, 1)
        else:
            x = self.rnn(inputs)
            assert isinstance(x, jax.Array)
            inputs = inputs.reshape(B, -1, 1)

        x = self.dense(x.reshape(-1, self.d_model))
        x = x.reshape(B, -1, 1)
        x = self.out_conv(x)

        return x + inputs



