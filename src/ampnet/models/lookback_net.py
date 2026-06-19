import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib

from ..F import apply_rnn_burn
from ..modules.gated import GLU
from ..modules.iir import expSmoother
from ..modules.misc import CausalLinear
from ..modules.activations import SnakeP


from typing import Any

Output = Any
Carry = tuple[jax.Array, jax.Array]


class LookbackCell(nnx.RNNCellBase):
    """
    sequential part of lookback net
    see: https://flax.readthedocs.io/en/v0.12.0/_modules/flax/nnx/nn/recurrent.html

    Expects:
        (B, T_block, block, d)
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,      # rnn hidden dim
        d_emb: int,        # block embed dim
        d_block: int,      # block size
        n_lookback: int    # sample lookback

    ) -> None:
        self.rnn = nnx.SimpleCell(
            in_features=d_emb,
            hidden_features=d_model,
            rngs=rngs
        )
        self.block_emb = nnx.Linear(
            in_features=d_block+n_lookback,
            out_features=d_emb,
            rngs=rngs
        )

        self.readout = nnx.Linear(
            in_features=d_model,
            out_features=d_block,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs
        )

        self.d_emb = d_emb
        self.n_lookback = n_lookback
        self.d_block = d_block
        self.act = nnx.silu

    def initialize_carry(
      self,
      input_shape: tuple[int, ...],
      rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ) -> Carry:
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: (B, feature_axis, ...)
            - i.e. time-axis removed

        Returns:
          An initialized carry for the given RNN cell.
        """
        B_dims = input_shape[:-self.num_feature_axes]
        block_size, d = input_shape[-self.num_feature_axes:]

        rnn_inp_shape = (*B_dims, self.d_emb)

        rnn_carry = self.rnn.initialize_carry(rnn_inp_shape, rngs)
        lb_cache = jnp.zeros((*B_dims, self.n_lookback))

        return (lb_cache, rnn_carry)

    def __call__(
        self,
        carry: Carry,
        inputs: jax.Array
    ) -> tuple[Carry, jax.Array]:
        """
        Run the RNN cell.

        Args:
          carry: the hidden state of the RNN cell.
          inputs: (B, d_block, 1)

        Returns:
          A tuple with the new carry and the output.
        """
        # TODO: consider non 1d blocks?
        B, *_ = inputs.shape

        lb_cache, rnn_carry = carry

        rnn_raw = jnp.concat([lb_cache, inputs.reshape(B, -1)], axis=1)
        rnn_in = self.block_emb(rnn_raw)

        rnn_carry, rnn_res = self.rnn(carry=rnn_carry, inputs=rnn_in)
        res = self.readout(rnn_res) + inputs.reshape(B, -1)
        res_detached = jax.lax.stop_gradient(res)

        # update rolling cache
        lb_cache = jnp.concat([lb_cache, res_detached], axis=1)[:, self.d_block:]

        return (lb_cache, rnn_carry), res

    @property
    def num_feature_axes(self) -> int:
        """
        Returns the number of feature axes of the RNN cell.

        USAGE NOTE:
        - nnx.RNN infers the left-immediate axis to features as time, i.e:
            (B1, B2, [this], feat1, feat2, ...)
        """

        # (B, T, block_dim, d)
        return 2


class LookbackCellO(LookbackCell):
    """
    variant of lookback cell that passes history to only output head
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,      # rnn hidden dim
        d_emb: int,        # block embed dim
        d_block: int,      # block size
        n_lookback: int    # sample lookback

    ) -> None:
        self.rnn = nnx.SimpleCell(
            in_features=d_emb,
            hidden_features=d_model,
            rngs=rngs
        )
        self.block_emb = nnx.Linear(
            in_features=d_block,
            out_features=d_emb,
            rngs=rngs
        )

        self.readout = nnx.Linear(
            in_features=d_model + n_lookback,
            out_features=d_block,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs
        )

        self.d_emb = d_emb
        self.n_lookback = n_lookback
        self.d_block = d_block

    def __call__(
        self,
        carry: Carry,
        inputs: jax.Array
    ) -> tuple[Carry, jax.Array]:
        """
        Run the RNN cell.

        Args:
          carry: the hidden state of the RNN cell.
          inputs: (B, d_block, 1)

        Returns:
          A tuple with the new carry and the output.
        """
        B, *_ = inputs.shape

        lb_cache, rnn_carry = carry
        rnn_in = self.block_emb(inputs.reshape(B, self.d_block))

        rnn_carry, rnn_res = self.rnn(carry=rnn_carry, inputs=rnn_in)

        readout_in = jnp.concat([rnn_res, lb_cache], axis=1)
        res = self.readout(readout_in) + inputs.reshape(B, -1)
        res_detached = jax.lax.stop_gradient(res)

        # update rolling cache
        lb_cache = jnp.concat([lb_cache, res_detached], axis=1)[:, self.d_block:]

        return (lb_cache, rnn_carry), res


class LookbackCellIO(LookbackCell):
    """
    variant of lookback cell that passes history to backbone
    and output head
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,      # rnn hidden dim
        d_emb: int,        # block embed dim
        d_block: int,      # block size
        n_lookback: int    # sample lookback

    ) -> None:
        self.d_emb = d_emb
        self.n_lookback = n_lookback
        self.d_block = d_block

        self.rnn = nnx.SimpleCell(
            in_features=d_emb,
            hidden_features=d_model,
            rngs=rngs
        )
        
        # MOD 2
        self.block_emb = CausalLinear(
            in_features=d_block + n_lookback,
            out_features=d_emb,
            rngs=rngs,
            axis=1
        )
        # self.block_emb = nnx.Linear(
        #     in_features=d_block + n_lookback,
        #     out_features=d_emb,
        #     rngs=rngs
        # )
        self.readout = nnx.Sequential(
            nnx.Linear(
                in_features = d_block + d_model + n_lookback,
                out_features = d_block,
                kernel_init=nnx.initializers.zeros_init(),
                rngs=rngs
            ),
            CausalLinear(
                in_features=d_block,
                out_features=d_block,
                rngs=rngs,
                axis=1
            )
        )


    def __call__(
        self,
        carry: Carry,
        inputs: jax.Array
    ) -> tuple[Carry, jax.Array]:
        """
        Run the RNN cell.

        Args:
          carry: the hidden state of the RNN cell.
          inputs: (B, d_block, 1)

        Returns:
          A tuple with the new carry and the output.
        """
        B, *_ = inputs.shape

        inputs = inputs.reshape(B, -1)
        lb_cache, rnn_carry = carry
        emb_in = jnp.concat([lb_cache, inputs], axis=1)
        rnn_in = self.block_emb(emb_in)

        rnn_carry, rnn_res = self.rnn(carry=rnn_carry, inputs=rnn_in)

        readout_in = jnp.concat([inputs, rnn_res, lb_cache], axis=1)
        res = self.readout(readout_in)

        # MOD
        # breakpoint()
        # res = jnp.concat([lb_cache[:, -1, None], res], axis=1)
        # res = jnp.cumsum(res, axis=1) #[:, :-1]
        # /MOD

        res = res + inputs

        res_detached = jax.lax.stop_gradient(res)

        # update rolling cache
        lb_cache = jnp.concat([lb_cache, res_detached], axis=1)[:, self.d_block:]

        return (lb_cache, rnn_carry), res

# --------------------------------------------------

class LookbackNet(nnx.Module):
    """
    chunk-based RNN with T-last generated outputs fed back as input features

    why:
        - BlockRNN has boundary artifacts despite theoretically keeping history
        - based off CarGAN and FarGAN: https://arxiv.org/abs/2110.10139
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,       # rnn hidden dim
        d_emb: int,         # block embed dim
        d_block: int,       # block size
        n_lookback: int,    # sample lookback
        variant: str = "I"

    ) -> None:
        self.d_model = d_model
        self.d_emb = d_emb
        self.d_block = d_block
        self.n_lookback = n_lookback

        if variant == "I":
            cell_class = LookbackCell
        elif variant == "O":
            cell_class = LookbackCellO
        elif variant == "IO":
            cell_class = LookbackCellIO
        else:
            raise ValueError(f"Unrecognized variant: {variant}")

        self.rnn = nnx.RNN(
            cell_class(
                rngs,
                d_model=d_model,
                d_emb=d_emb,
                d_block=d_block,
                n_lookback=n_lookback,
            )
        )

    def __call__(self, inputs: jax.Array, burn_steps=0):
        """
        inputs: (B, T, 1)
        """

        assert inputs.ndim == 3

        # time pad
        B, T, _ = inputs.shape
        block_pad = (-T) % self.d_block
        inputs_padded = jnp.pad(
            inputs,
            (
                (0, 0),
                (0, block_pad),
                (0, 0)
            )
        )
        T_new = inputs_padded.shape[1]
        blocks = inputs_padded.reshape(B, T_new // self.d_block, self.d_block, 1)

        if burn_steps > 0:
            block_burn_steps = burn_steps // self.d_block
            blocks_out = apply_rnn_burn(self.rnn, blocks, burn_steps=block_burn_steps)
        else:
            blocks_out = self.rnn(blocks)

        res = blocks_out.reshape(B, T_new, 1)[:, burn_steps:T, :]
        return res


