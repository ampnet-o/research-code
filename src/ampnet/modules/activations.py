import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib

from typing import Any


def snake(x: jax.Array):
    periodic = jnp.sin(x)
    return x + periodic*periodic


class SnakeP(nnx.Module):
    """
    Parameterized snake
    https://arxiv.org/abs/2006.08195
    """
    def __init__(
        self,
        n_units: int,
        rngs: nnx.Rngs
    ) -> None:
        self.a = nnx.Param(rngs.Param.normal((1, n_units,))/2 + 0.5)


    def __call__(self, x: jax.Array):
        """
        apply to last dimension

        x: (B1, B2, ..., d)
        """
        *B, d = x.shape
        x = x.reshape(-1, d)
        frac = 1/(2*self.a)
        periodic = frac*(-jnp.cos(2*self.a*x) + 1)

        res = x + periodic
        res = res.reshape(*B, d)
        return res


