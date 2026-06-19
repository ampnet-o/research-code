import jax.numpy as jnp
from flax import nnx

from ampnet.models.lookback_net import LookbackCell, LookbackCellIO, LookbackCellO, LookbackNet


def test_lookback_cell():

    h = nnx.RNN(
        LookbackCell(
            nnx.Rngs(0),
            d_model=16,
            d_emb=4,
            d_block=8,
            n_lookback=8,
        )
    )

    # (B, T_block, block, d)
    x = jnp.ones((1, 4096, 8, 1))
    print(h(x))


def test_lookback_o_cell():

    h = nnx.RNN(
        LookbackCellO(
            nnx.Rngs(0),
            d_model=16,
            d_emb=4,
            d_block=8,
            n_lookback=8,
        )
    )

    # (B, T_block, block, d)
    x = jnp.ones((1, 4096, 8, 1))
    print(h(x))


def test_lookback_io_cell():

    h = nnx.RNN(
        LookbackCellIO(
            nnx.Rngs(0),
            d_model=16,
            d_emb=4,
            d_block=8,
            n_lookback=8,
        )
    )

    # (B, T_block, block, d)
    x = jnp.ones((1, 4096, 8, 1))
    print(h(x))


def test_lookback_net():

    h = LookbackNet(
        nnx.Rngs(0),
        d_model=32,
        d_emb=4,
        d_block=8,
        n_lookback=15
    )
    # (B, T_block, block, d)
    x = jnp.ones((1, 4099, 1))
    print(h(x))


def test_lookback_net_burnin():

    burn = 1024
    h = LookbackNet(
        nnx.Rngs(0),
        d_model=32,
        d_emb=4,
        d_block=8,
        n_lookback=15
    )
    # (B, T_block, block, d)
    x = jnp.ones((1, 4099, 1))
    print(h(x, burn_steps=burn))



def test_lookback_net_o():

    h = LookbackNet(
        nnx.Rngs(0),
        d_model=32,
        d_emb=4,
        d_block=8,
        n_lookback=15,
        variant="O"
    )
    # (B, T_block, block, d)
    x = jnp.ones((1, 4099, 1))
    print(h(x))


def test_lookback_net_o_burnin():

    burn = 1024
    h = LookbackNet(
        nnx.Rngs(0),
        d_model=32,
        d_emb=4,
        d_block=8,
        n_lookback=15,
        variant="O"
    )
    # (B, T_block, block, d)
    x = jnp.ones((1, 4099, 1))
    print(h(x, burn_steps=burn))

