# temporal convolutional networks
# wavenet and more

import jax
import jax.numpy as jnp
from flax import nnx

from typing import Callable

class TCNBaselineBlock(nnx.Module):
    """
    Block for temporal convolutional network as per https://arxiv.org/abs/1803.01271

    Features
    - causal conv (i.e. left-padded)
    * weight norm (weight parameterized as scale*direction)
    - relu
    * dropout
    - residual block
    """
    def __init__(
        self,
        kernel_size: int,
        d_inner: int,
        d_input: int,
        rngs: nnx.Rngs,
        dilation: int=1,
        act: Callable = nnx.swish
    ) -> None:

        # Unsure:
        # - 1x1 conv?

        # time-wise conv
        self.up_conv = nnx.WeightNorm(
            nnx.Conv(
                in_features=d_input,
                out_features=d_inner,
                kernel_dilation=dilation,
                padding="CAUSAL",
                kernel_size=kernel_size,
                rngs=rngs
            ),
            rngs=rngs
        )

        # feature-wise conv
        self.down_conv = nnx.WeightNorm(
            nnx.Conv(
                in_features=d_inner,
                out_features=d_input,
                kernel_size=1,
                rngs=rngs
            ),
            rngs=rngs
        )
        # self.act = nnx.swish
        self.act = nnx.tanh

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        inputs: (B, T, d)
        """
        x = self.up_conv(inputs)
        x = self.act(x)
        x = self.down_conv(x)
        x = self.act(x)

        return x + inputs


class TCNBaseline(nnx.Module):
    """
    temporal convolutional network as per https://arxiv.org/abs/1803.01271
    """

    def __init__(
        self,
        kernel_size: int,
        d_model: int,
        d_up: int,
        n_layers: int,
        rngs: nnx.Rngs,
        dilation_scale: int=2,
    ):
        self.backbone_in = nnx.Conv(
            in_features=1,
            out_features=d_model,
            padding="CAUSAL",
            kernel_size=1,
            rngs=rngs
        )
        self.backbone = nnx.Sequential(*[
            TCNBaselineBlock(
                kernel_size=kernel_size,
                d_inner=d_up,
                d_input=d_model,
                dilation=max(1, i*dilation_scale),
                rngs=rngs
            )
        for i in range(n_layers)])

        self.backbone_out = nnx.Conv(
            in_features=d_model,
            out_features=1,
            padding="CAUSAL",
            kernel_size=1,
            rngs=rngs
        )

    def __call__(self, inputs: jax.Array):
        x = self.backbone_in(inputs)
        x = self.backbone(x)
        x = self.backbone_out(x)

        return x + inputs


# --------------------------------------------------


class TCNSynth(nnx.Module):
    """
    temporal convolutional network as per https://arxiv.org/abs/1803.01271
    """

    def __init__(
        self,
        kernel_size: int,
        d_model: int,
        d_up: int,
        n_layers: int,
        rngs: nnx.Rngs,
        dilation_scale: int=2,
    ):
        self.backbone_in = nnx.Conv(
            in_features=1,
            out_features=d_model,
            padding="CAUSAL",
            kernel_size=1,
            rngs=rngs
        )
        self.backbone = nnx.Sequential(*[
            TCNBaselineBlock(
                kernel_size=kernel_size,
                d_inner=d_up,
                d_input=d_model,
                dilation=max(1, i*dilation_scale),
                rngs=rngs
            )
        for i in range(n_layers)])

        self.backbone_out = nnx.Conv(
            in_features=d_model,
            out_features=1,
            padding="CAUSAL",
            kernel_size=1,
            rngs=rngs
        )

    def __call__(self, inputs: jax.Array):
        x = self.backbone_in(inputs)
        x = self.backbone(x)
        x = self.backbone_out(x)

        return x + inputs

