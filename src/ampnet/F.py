import jax
import jax.numpy as jnp
from flax import nnx

def pre_emphasis(x: jax.Array, scale=0.95):
    """
    Apply pre-emphasis filter to time series of shape
    (B, T, d)

    NOTE: uses zero padding
    """
    x = x - scale*jnp.roll(x, 1, axis=1)
    return x.at[:, 0, :].set(0)


def downsample(x: jax.Array, scale = 2, rate: int = 44100):
    """
    x: (B, T, 1)
    """
    fx = jnp.fft.fft(x)
    pass

def apply_rnn_burn(model: nnx.RNN, input: jax.Array, burn_steps: int):
    """
    for autodiff purposes only

    input: (B, T, d, ...)
    """

    B, T, *_ = input.shape

    carry, pre = jax.lax.stop_gradient(model.__call__(
        input[:, :burn_steps],
        return_carry=True
    ))

    # carry = carry[:, -1, :].reshape(B, -1)
    post_inp = input[:, burn_steps:]
    post = model.__call__(
        post_inp,
        initial_carry=carry,
        return_carry=False
    )
    assert isinstance(post, jax.Array)
    assert isinstance(pre, jax.Array)

    return jnp.concat([pre, post], axis=1)


