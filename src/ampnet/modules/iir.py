import jax
import jax.numpy as jnp

import flax
from flax import nnx
from flax.nnx import rnglib

from ..F import apply_rnn_burn
from ..modules.gated import GLU

from typing import Any


def exp_smooth(y0: jax.Array, x: jax.Array, alpha=0.85):
    """
    computes: y[t] = x[t] + ay[t-1]

    y0: (B, 1)
    x: (B, T, 1)
    alpha: in (0, 1)
    """
    pass


class expSmoother(nnx.Module):
    """
    naively computes: y[t] = x[t] + ay[t-1]
    by convolution

    a in (0, 1)
    """

    def __init__(
        self,
        xlen: int,
        alpha: float = 0.85,
    ) -> None:

        assert 0 < alpha < 1
        assert xlen > 2

        self._lpad = xlen-2
        self.kern = jnp.array([alpha**i for i in reversed(range(xlen))]).reshape(-1, 1, 1)

    def __call__(self, y0: jax.Array, x: jax.Array) -> jax.Array:
        """
        y0: (B, 1)
        x: (B, T, 1)
        """

        B = y0.shape[0]
        y0 = y0.reshape((B, 1, 1))
        conv_mat = jnp.concatenate([y0, x], axis=1)

        dn = jax.lax.conv_dimension_numbers(
            conv_mat.shape,
            self.kern.shape,
            (
                "NWC",    # dat labels
                "WIO",    # kern labels
                "NWC"     # out labels
            )
        )

        res = jax.lax.conv_general_dilated(
            conv_mat,
            self.kern,
            window_strides=(1,),
            padding=[(self._lpad, 0)],
            lhs_dilation=(1,),
            rhs_dilation=(1,),
            dimension_numbers=dn
        )

        return res


