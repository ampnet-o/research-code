import numpy as np
import jax
import jax.numpy as jnp
import librosa

import math


def lfilter(b, a, x, y_ic):
    """
    iir linear filter along last axis, with initial conditions y_hist

    by rule (assuming y = output signal):
        a[0]*y[n] =
        b[0]*x[n] + ... + b[M]x[n-M]
    """
    pass


def pad_time(x: jax.Array, pad_amount: int):
    """
    utility function

    x: (B, T, d)
    """
    return jnp.pad(
        x,
        (
            (0, 0),            # 0 + b + 0
            (0, pad_amount),   # 0 + t + pad_amount
            (0, 0)             # 0 + d + pad_amount
        )

    )


def init_legendre(size: int, deg: int):
    """
    pointwise eval of legendre polys as per recurrence:
        (n+1)P_{n+1}(x) = (2n+1)xP_n(x)-nP_{n-1}(x)
    NOTE: try keeping this small

    return: (t_block, n_deg)
    """
    assert deg > 1
    inps = np.linspace(start=-1, stop=1, num=size)
    res = np.ones((deg, size), dtype=np.float32)
    res[1] = inps

    for n in range(1, deg-1):
        res[n+1] = (2*n + 1)*res[n]*inps - n*res[n-1]

    return jnp.array(res).T


def apply_window(window: jax.Array, signal: jax.Array, hop_size=2):
    """
    jax impl of apply window
    signal: (B, T)
    """
    assert window.ndim == 1
    assert signal.ndim == 2

    B, T = signal.shape
    signal_reshaped = signal.reshape(B, 1, T)

    # Kernel: [out_channels, in_channels, kernel_length]
    kernel = jnp.diag(window)[:, None, :]  # shape: (window_size, 1, window_size)
    result = jax.lax.conv_general_dilated(
        signal_reshaped,
        kernel,
        window_strides=(hop_size,),
        padding='VALID',
        dimension_numbers=('NCH', 'OIH', 'NCH')
    )

    # Result shape: (B, window_size, num_windows)
    windowed_signal = jnp.transpose(result, (0, 2, 1))
    return windowed_signal


_apply_window_cpu_impl = jax.jit(
    apply_window,
    backend='cpu',
    static_argnums=(2,)  # hop size
)

def apply_window_cpu(window, signal, hop_size: int) -> np.ndarray:
    """
    cpu version of apply_window. To be used in pre-processing
    """
    return _apply_window_cpu_impl(window, signal, hop_size=hop_size)


def overlap_add(frames: jax.Array, hop: int):
    """
    frames: (*B, F, d)
    """

    assert hop > 0
    assert frames.ndim > 2

    *B, n_frames, d_frame = frames.shape
    B_flat = math.prod(B)
    stack = frames.reshape(-1, n_frames, d_frame)
    expected_len = (n_frames - 1) * hop + d_frame


    hop_pad = (-d_frame) % hop              # negative mod measures overshoot
    d_frame_ext = d_frame + hop_pad         # frame size in chunk aligned case
    group_size = d_frame_ext // hop         # no. overlapping chunks / frame
    frame_pad = (-n_frames) % group_size    # pad frames to whole chunks

    # group by max overlapping frames
    stack = jnp.pad(
        stack,
        (
            (0, 0),                       # B_flat
            (0, frame_pad + group_size),  # F
            (0, hop_pad)                  # d
        )
    )
    n_frames_ext = stack.shape[1]
    n_framegroups = n_frames_ext // group_size

    # -- ext space --

    # concat non-overlapping groups
    stack = stack.reshape(B_flat, n_framegroups, group_size, -1)
    joined = stack.transpose(0, 2, 1, 3).reshape(B_flat, -1)

    # shift boundaries for zero spillover to impl np.roll
    trunc_shape = (group_size, n_framegroups*d_frame_ext - hop)
    joined = joined[:, :math.prod(trunc_shape)].reshape(B_flat, *trunc_shape)
    res = jnp.sum(joined, axis=1)

    # trim redundant samples from hop group reshape trick
    res = res[:, :expected_len]
    return res.reshape(*B, *res.shape[1:])


# --------------------------------------------------


def mu_law_enc(x: jax.Array, mu=255):
    """
    mu law companding

    x: (N, )
    """
    sign = jnp.sign(x)
    div = jnp.log(1 + mu)
    return sign * (jnp.log(1 + mu*jnp.abs(x)) / div)

def mu_law_dec(y: jax.Array, mu=255):
    """
    y: (N, )
    """
    sign = jnp.sign(y)
    return sign * (jnp.pow(1+mu, jnp.abs(y)) / mu)


def mu_law_q_enc(x: jax.Array, classes=256, mu=255):
    """
    Mu-law encode and quantize
    """
    bins = jnp.linspace(-1, 1, num=classes)  # inefficient, but it's tiny
    enc = mu_law_enc(x, mu=mu)

    return jnp.digitize(enc, bins)


def mu_law_q_dec(x_ids: jax.Array, classes=256, mu=255):
    scale = 2 / (classes - 1)
    x = scale*x_ids - 1
    return mu_law_dec(x, mu=mu)


# --------------------------------------------------

def SNR(y_pred, y, eps=1e-6):
    """
    Signal to noise ratio in db

    i.e Var(y-y_pred) / Var(y)
    """
    ratio = (jnp.var(y) + eps) / jnp.var(y_pred - y)
    return 20*jnp.log10(ratio)


# --------------------------------------------------

def lpc_windowed(
    signal: jax.Array,
    order: int
):
    """
    apply lpc on single window
    """

    coeffs = librosa.lpc(
        signal,
        order = order

    )
    pass

def lpc_windowed(
    signal: np.ndarray,
    window: np.ndarray,
    hop_size: int,
    order: int,
    combine_method: str = "linterp"
):
    """
    Apply lpc on windowed version of signal

    linterp = linear scale, overlap-add
    nn = nearest neighbor interpolation of coeffs

    expect:
        signal: (B, T,)
    """
    assert window.ndim == 1
    window_len = window.shape[0]

    frames = np.array(apply_window_cpu(window, signal, hop_size))
    B, N, d = frames.shape           # (B, N_frames, d_frame)
    frames = frames.reshape(B*N, d)
    coeffs = librosa.lpc(            # (B*N_frames, order+1)
        frames,
        order=order,
        axis=1
    )

    overlap_amount = (window_len - hop_size) // 2
    # TODO: linterp method
    # this is the knn method

    window_tiled = np.tile(frames, overlap_amount)

    return coeffs




