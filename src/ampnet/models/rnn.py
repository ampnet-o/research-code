import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import RNN

from ..F import apply_rnn_burn
from ..modules.fourier_embed import SinEmbed

from typing import Type

class BaselineRNN(nnx.Module):
    """
    Basic RNN with burn-in, i.e. warmup
    """
    def __init__(
        self,
        d_model: int,
        rngs: nnx.Rngs,
        in_features: int=1,
        blocks: int=0,
        activation_fn = nnx.gelu,
        rnn_type: Type[nnx.RNNCellBase] = nnx.SimpleCell
    ) -> None:
        """
        Initialize a RNN with hidden dimension d_model
        """

        self.rnn_in = RNN(
            nnx.SimpleCell(
                in_features=in_features,
                hidden_features=d_model,
                rngs=rngs
            ),
            return_carry=False,
        )

        self.rnn_stack = nnx.data([
            nnx.RNN(nnx.SimpleCell(d_model, d_model, rngs=rngs))
            for _ in range(blocks)
        ])
        self.ff_stack = nnx.data([nnx.Sequential(
            activation_fn,
            nnx.Linear(d_model, 1, rngs=rngs)
        ) for _ in range(blocks)])
        self.blocks = blocks

        self.out_proj = nnx.Linear(in_features=d_model, out_features=1, rngs=rngs)
        self.d_model = d_model

    def __call__(self, inputs: jax.Array, burn_steps: int=0):
        """
        NOTE: burn_steps should be static. changing will trigger recompilation
        """
        if burn_steps > 0:
            return self._impl_call_with_carry(inputs, burn_steps)

        return self._impl_call(inputs)

    def _impl_call_with_carry(self, inputs: jax.Array, burn_steps: int):
        h = apply_rnn_burn(self.rnn_in, inputs, burn_steps)
        h += inputs

        for i in range(self.blocks):
            h_res = apply_rnn_burn(self.rnn_stack[i], h, burn_steps)
            h_res = self.ff_stack[i](h_res)
            h += h_res

        return self.out_proj(h[:, burn_steps:])

    def _impl_call(self, inputs: jax.Array):
        h = self.rnn_in(inputs)
        assert isinstance(h, jax.Array)
        h += inputs

        for i in range(self.blocks):
            h_res = self.rnn_stack[i](h)
            assert isinstance(h_res, jax.Array)
            h_res = self.ff_stack[i](h_res)
            h += h_res

        return self.out_proj(h)

class RNNV2_t(nnx.Module):
    """
    Basic RNN but with absolute time embed

    why:
        - Vanilla RNN has trouble with high freq features
    """
    def __init__(
        self,
        rngs,
        d_model: int=128,
        d_embed: int=128,
        t_quanta: float=0.2
    ):
        self.rnn = RNN(
            nnx.SimpleCell(
                in_features=d_embed,
                hidden_features=d_model,
                rngs=rngs
            ),
            return_carry=False,
        )

        assert d_embed % 2 == 0

        self.dense = nnx.Linear(in_features=d_model, out_features=1, rngs=rngs)
        self.d_model = d_model
        self.t_embed = SinEmbed(
            (0, 8192),
            d_embed//2,
        )
        self.t_quanta = t_quanta

    def __call__(self, inputs: jax.Array, burn_steps: int=0):
        """
        inputs: (B, T, 1)
        """
        x = self.apply_embed(inputs)
        if burn_steps > 0:
            x = apply_rnn_burn(self.rnn, x, burn_steps)[:, burn_steps:]
            inputs = inputs[:, burn_steps:]
        else:
            x = self.rnn(x)
            assert isinstance(x, jax.Array)

        return self.dense(x) + inputs

    def apply_embed(self, inputs: jax.Array):
        B, T, _ = inputs.shape
        t_embed = self.t_embed(jnp.arange(T)*self.t_quanta).reshape(1, T, -1)
        x = t_embed*inputs
        return x


class RNNV2(nnx.Module):
    """
    Basic RNN but with intensity embed
    """
    def __init__(
        self,
        rngs,
        d_model: int=128,
        d_embed: int=128,
    ):
        self.rnn = RNN(
            nnx.SimpleCell(
                in_features=d_embed,
                hidden_features=d_model,
                rngs=rngs
            ),
            return_carry=False,
        )

        assert d_embed % 2 == 0

        self.dense = nnx.Linear(in_features=d_model, out_features=1, rngs=rngs)
        self.d_model = d_model
        self.i_embed = SinEmbed(
            (-32768, 32767),
            d_embed//2,
        )

    def __call__(self, inputs: jax.Array, burn_steps: int=0):
        """
        inputs: (B, T, 1)
        """
        B, T, _ = inputs.shape

        if burn_steps > 0:
            x = self.i_embed(inputs.flatten()).reshape(B, T, -1)
            _, x = apply_rnn_burn(self.rnn, x, burn_steps)
            inputs = inputs[:, burn_steps:]
        else:
            x = self.i_embed(inputs.flatten())
            x = self.rnn(x.reshape(B, T, -1))
            assert isinstance(x, jax.Array)

        return self.dense(x) + inputs


class MuNet(nnx.Module):
    """
    Basic single-sample RNN with mu-law embedding
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int=128,
        n_embed: int=256
    ):

        self.n_embed = n_embed
        self.d_model = d_model

        self.embed = nnx.Embed(
            num_embeddings=n_embed,
            features=d_model,
            rngs=rngs,
        )
        self.rnn = RNN(
            nnx.SimpleCell(
                in_features=d_model,
                hidden_features=d_model,
                rngs=rngs
            ),
            return_carry=False,
        )

        self.readout = nnx.Linear(
            in_features=d_model,
            out_features=n_embed,
            rngs=rngs
        )


    def __call__(self, inputs: jax.Array, burn_steps: int=0):
        """
        NOTE: burn_steps should be static. changing will trigger recompilation

        expect:
            inputs: (B, T, 1)  [uint]
        """
        B, T, _ = inputs.shape
        inp_embedded = self.embed(inputs).reshape(B, T, self.d_model)

        if burn_steps > 0:
            res = apply_rnn_burn(self.rnn, inp_embedded, burn_steps)[:, burn_steps:]
            inp_embedded = inp_embedded[:, burn_steps:]
        else:
            res = self.rnn(inp_embedded)

        x = res + inp_embedded
        x = self.readout(x)

        return x.reshape(B, -1, self.n_embed)

    def sample(self, inputs: jax.Array):
        """
        Greedy sampling
        """
        logits = self.__call__(inputs, burn_steps=0)   # (B, T, n_bins)
        probs = jax.nn.softmax(logits, axis=2)
        return jnp.argmax(probs, axis=2)




