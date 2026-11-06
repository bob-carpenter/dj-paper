"""Simple wrappers around Jax's built-in density functions"""

import jax.numpy as jnp
from jax.scipy import stats
from jax import random

from .util import broadcast_shapes_of


def normal(y, mu, sigma):
    return jnp.sum(stats.norm.logpdf(y, mu, sigma))


def normal_rng(rng, mu, sigma):
    return mu + sigma * random.normal(rng, shape=broadcast_shapes_of(mu, sigma))


def exponential(y, beta):
    return jnp.sum(stats.expon.logpdf(y, beta))
