# from test_lookbacknet import test_lookback_io_cell as foo
from test_modules import test_glu_simple as foo

from ampnet.modules.misc import CausalLinear

from typing import Callable
import time
import os

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

jax.config.update('jax_platform_name', 'cpu')


b = CausalLinear(3, 16, nnx.Rngs(0), axis=1)

x = jnp.ones([17, 3, 8])

print(b(x).shape)

