import jax
import jax.numpy as jnp
from flax import nnx


class LegendreEmbed(nnx.Module):
    """
    Legendre Polynomial regression embedding of an audio block
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_query: int,
        d_embed: int,
        n_codewords: int,
    ) -> None:
        self.n_codewords = n_codewords
        self.d_embed = d_embed
        self.W_K = nnx.Param(rngs.params.normal((self.n_codewords, d_query)))
        self.codebook = nnx.Param(rngs.params.normal((self.n_codewords, d_embed)))



    def __call__(self, x: jax.Array):
        """
        x: (B, T, d_query)
        """
        # TRY: abs, put sign into encoding?
        sim = jnp.einsum("BTq,Nq->BTN", x, self.W_K)
        ids = jnp.argmax(sim, axis=2)   # (B, T, 1)


class PolyEmbed(nnx.Module):
    """
    Polynomial regression embedding of an audio block
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_query: int,
        d_embed: int,
        n_codewords: int,
    ) -> None:
        self.n_codewords = n_codewords
        self.d_embed = d_embed
        self.W_K = nnx.Param(rngs.params.normal((self.n_codewords, d_query)))
        self.codebook = nnx.Param(rngs.params.normal((self.n_codewords, d_embed)))



    def __call__(self, x: jax.Array):
        """
        x: (B, T, d_query)
        """
        # TRY: abs, put sign into encoding?
        sim = jnp.einsum("BTq,Nq->BTN", x, self.W_K)
        ids = jnp.argmax(sim, axis=2)   # (B, T, 1)


class VQEmbed(nnx.Module):
    """
    VQ embedding of an audio block
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_query: int,
        d_embed: int,
        n_codewords: int,
    ) -> None:
        self.n_codewords = n_codewords
        self.d_embed = d_embed
        self.W_K = nnx.Param(rngs.params.normal((self.n_codewords, d_query)))
        self.codebook = nnx.Param(rngs.params.normal((self.n_codewords, d_embed)))



    def __call__(self, x: jax.Array):
        """
        x: (B, T, d_query)
        """
        # TRY: abs, put sign into encoding?
        sim = jnp.einsum("BTq,Nq->BTN", x, self.W_K)
        ids = jnp.argmax(sim, axis=2)   # (B, T, 1)



class ConvEmbed(nnx.Module):
    """
    conv-transpose-conv encoding of an audio block
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_embed: int,
    ) -> None:
        self.d_embed = d_embed
        self.filters = nnx.Param

        nnx.Conv


    def __call__(self, x: jax.Array):
        """
        x: (B, 1) or (B, )
        """
        pass


class LinConvEmbed(nnx.Module):
    """
    conv-transpose-conv encoding of an audio block
    but kernels are all linear (slop + y-offset)
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_embed: int,
    ) -> None:
        self.d_embed = d_embed
        self.filters = nnx.Param

        nnx.Conv


    def __call__(self, x: jax.Array):
        """
        x: (B, 1) or (B, )
        """
        pass

