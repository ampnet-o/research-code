import jax
import jax.numpy as jnp


# doesn't seem to be faster than nnx conv
def fast_1d_cconv(
    x: jax.Array,
    kernel: jax.Array,
    dilation: int,
):
    """
    faster 1d causal convolution for wavenets.
    expects many small kernels

    x: (B, T, d)
    y: (F, k)

    Return:
        (B, T, F, d)
    """

    # TODO: parameterize axis for convolution (currently assumes t)
    # TODO^2: this might not be necessary

    B, T, d = x.shape
    F, k = kernel.shape
    delay_pad = k*dilation

    # TODO: check the correctness of these
    new_idx = T + ((k-1)*dilation)
    trunc_idx = k*new_idx


    # offset trick
    # (B, delay_copies, T, d)
    x_stack = jnp.stack(
        [x for _ in range(k)],
        axis=1
    )
    x_padded = jnp.pad(
        x_stack,
        (
            (0, 0),             # B
            (0, 0),             # delay_copies
            (0, delay_pad),     # T
            (0, 0),             # D
        )
    )
    x_padded = x_padded.reshape(B, -1, d)
    x_padded = x_padded[:, :trunc_idx, :]
    x_shifted = x_padded.reshape(B, k, new_idx, d)

    # (B, k, T, d)
    x_delayed = x_shifted[:, :, :T, :]
    res = jnp.einsum("BKTD,FK->BTFD", x_delayed, kernel)

    return res

