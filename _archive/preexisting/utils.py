import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, vmap, jacfwd
import scipy as sp
from scipy import fft, special
import numpy as np
#import orthax as ox
import math as mt
import scipy.linalg as scla

def gridpts(N, with_weights=False):
    x = jnp.cos(jnp.pi * (2 * jnp.arange(N) + 1) / (2 * N))
    if with_weights:
        weights = (2 / N) * np.ones_like(x)
        return x, weights
    else:
        return x


