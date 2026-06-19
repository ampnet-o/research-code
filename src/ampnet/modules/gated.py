import jax
import jax.numpy as jnp
from flax import nnx

from typing import Callable

class GLU(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        act: Callable[[jax.Array], jax.Array] = nnx.sigmoid,
    ) -> None:
        """
        Gated linear unit on last axis of input
        """
        self.in_features = in_features
        self.out_features = out_features
        self.act = act

        # impl specific
        self.linear1 = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs
        )

        self.linear2 = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros_init()      # TODO: parameterize this
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        inputs: (*B, d)
        """
        # *B, d = inputs.shape
        up = self.linear2(inputs)
        up2 = self.act(self.linear2(inputs))
        return up * up2
