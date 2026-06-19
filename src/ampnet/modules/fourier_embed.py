import jax
import jax.numpy as jnp
from flax import nnx


class FourierEmbed(nnx.Module):
    """
    Fourier feature representation of a real number
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_embed: int,
        scale: float,
        init: str="normal"
    ) -> None:

        if init == "uniform":
            self.B = scale*rngs.uniform((d_embed,))
        else:
            self.B = scale*rngs.normal((d_embed,))

    def __call__(self, x: jax.Array):
        """
        x: (B, 1) or (B, )
        """
        bx = self.B @ x
        return jnp.concat([jnp.sin(bx), jnp.cos(bx)], axis=0)


# --------------------------------------------------


class SinEmbed(nnx.Module):
    """
    sinPE representation of a real number range
    """

    d_embed: int

    def __init__(
        self,
        range: tuple[float, float],
        d_inner: int,                # intrinsic dim
        f_max: float = 16384         # max materialized internal frequency
    ) -> None:

        r_min, r_max = range
        assert r_min < r_max
        assert d_inner > 0

        self._r_min = r_min
        self._r_max = r_max
        self.d_inner = d_inner
        self.d_embed = d_inner*2
        self._f_max = f_max

        # NOTE: generally, don't want freqs to cause overflow
        # max 4 zeros as a guess for now
        r_range = r_max - r_min
        self._scale = self._f_max / r_range

        self.freqs = 2*jnp.pi*jnp.reciprocal(jnp.exp(
            (jnp.arange(d_inner)/d_inner) * jnp.log(self._f_max)
        ))


    def __call__(
        self,
        inputs: jax.Array,
        mode: str = "clamp"
    ):
        """
        inputs: (B, 1) or (B, )
        """
        B = inputs.shape[0]
        inputs = self._impl_sin_inputs(inputs, mode, B)
        x, y = self._impl_apply_sines(inputs)
        return self._impl_combine_sines(x, y, B)

    # --------------------------------------------------

    def _impl_sin_inputs(
        self,
        inputs: jax.Array,
        mode: str,
        B: int
    ):
        inputs = inputs.reshape(B, 1)

        if mode == "clamp":
            inputs = jnp.clip(inputs, self._r_min, self._r_max)

        offset = -self._r_min
        inputs = (inputs + offset) * self._scale
        return inputs

    def _impl_apply_sines(
        self,
        inputs: jax.Array
    ):

        x = jnp.cos(inputs*self.freqs)[..., None]
        y = jnp.sin(inputs*self.freqs)[..., None]

        return x, y

    def _impl_combine_sines(
        self,
        x_cos: jax.Array,
        y_cos: jax.Array,
        B: int
    ):
        return jnp.concat([x_cos, y_cos], axis=2).reshape(B, -1)




