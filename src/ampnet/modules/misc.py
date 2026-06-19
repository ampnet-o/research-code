import jax
import jax.numpy as jnp
from flax import nnx

import math

class CausalLinear(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        axis: int = -1,
    ) -> None:
        """
        Linear transform with causal mask. Default to last axis
        """
        self.in_features = in_features
        self.out_features = out_features
        self.axis = axis

        self.W = nnx.Param(
            rngs.params.normal((self.out_features, self.in_features))
        )


    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        inputs: (*B, d_in, *d)

        return:
            (*B, d_out, *d)
        """

        # normalize index
        ax = len(inputs.shape) + self.axis if self.axis < 0 else self.axis

        # TODO: check if there's aliases at boundaries
        B = inputs.shape[:ax]
        d_i = inputs.shape[ax]
        d = inputs.shape[ax + 1:]

        masked = jnp.tril(self.W)
        inputs = inputs.reshape(math.prod(B), d_i, math.prod(d))
        res = jnp.einsum("BID,OI->BOD", inputs, masked)

        return res.reshape(*B, self.out_features, *d)


