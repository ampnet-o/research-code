import jax.numpy as jnp
from flax import nnx

from ampnet.models.rnn import MuNet

# MuNet test
def test_munet_basic():
    h = MuNet(
        nnx.Rngs(0),
        d_model=64,
        n_embed=256
    )
    print(h(jnp.ones((16, 100, 1), dtype=jnp.uint16)).shape)
    print(h(jnp.ones((16, 100, 1), dtype=jnp.uint16), burn_steps=13).shape)


