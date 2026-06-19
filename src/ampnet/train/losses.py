import jax
import jax.numpy as jnp

from ..F import pre_emphasis

def ESR_loss(y_pred, y, eps=1e-6):
    """
    Vanilla error-signal ratio loss

    i.e Var(y-y_pred) / Var(y)

    expect:
        y_pred: (B, T, 1)
        y: (B, T, 1)
    """
    # B, T, _ = y.shape
    return jnp.var(y_pred - y)/(jnp.var(y) + eps)


def pre_emp_ESR_loss(y_pred, y, eps=1e-6):
    """
    ESR loss with pre-emphasis filter

    i.e Var(y-y_pred) / Var(y)

    expect:
        y_pred: (B, T, 1)
        y: (B, T, 1)
    """
    B, T, _ = y.shape
    y_pred = pre_emphasis(y_pred)
    y = pre_emphasis(y)
    return jnp.var(y_pred - y)/(jnp.var(y) + eps)


def MSS_loss(y_pred, y):
    """
    multi-scale spectral loss

    https://ieeexplore.ieee.org/document/10319088
    """
    pass


