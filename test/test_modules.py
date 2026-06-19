from ampnet.modules.fourier_embed import SinEmbed
from ampnet.modules.gated import GLU

import jax
import jax.numpy as jnp
from flax import nnx


def test_sin_embed_simple():
    """
    Most basic sanity test
    """
    emb = SinEmbed(
        (-32000, 32000),
        128,
    )

    x = emb(jnp.array([1, 10, 100, 1000, 10000, 30000]))
    assert x.shape == (6, 256)
    assert jnp.all(jnp.abs(x) <= 1)


def test_glu_simple():
    glu = GLU(
        in_features=8,
        out_features=8,
        rngs=nnx.Rngs(0)
    )

    res = glu(jnp.ones((5, 8)))
    assert res.shape == (5, 8)


