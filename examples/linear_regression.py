from jax import random, jit
import jax.numpy as jnp
from jax.scipy import stats

from densityjax import (
    make_log_density,
    positive,
    real,
)
from densityjax.constraints import constrain, spec_to_pytree
from densityjax.util import ravelize_function

# These are the primary exports of this module:
__all__ = [
    "log_density",
    "log_density_vec",
    "constraints",
    "generated_quantities",
    "generated_quantities_vec",
]


# data is closed over, so we need some fake data
x = jnp.array([[1.0, 2.0, 3.0], [0.2, 0.1, 0.4]]).T  # 3x2
y = jnp.array([2.1, 3.7, 6.5])


## Parameter definitions

# This can be partial (or even entirely omitted) if you do not require any reshaping utilities
#   (ravelize_function, spec_to_pytree, init_random, etc.),
# otherwise it must cover all parameters to get the shapes/dtypes correct.
parameter_spec = {
    "alpha": real(),
    "beta": real(shape=x.shape[1]),
    "sigma": positive(),
}


def log_likelihood(alpha, beta, sigma):
    # prior
    target = jnp.sum(stats.norm.logpdf(alpha, loc=0.0, scale=1.0))
    target += jnp.sum(stats.norm.logpdf(beta, loc=0.0, scale=1.0))
    # "scale" is rate of exponential distribution (bad SciPy)
    target += jnp.sum(stats.expon.logpdf(sigma, scale=1.0))
    # likelihood
    mu = alpha + x @ beta
    target += jnp.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    return target


# a log density function
log_density = make_log_density(log_likelihood, parameter_spec=parameter_spec)


# We can also provide a flattened version, automatically,
# using the structure of the parameters defined above.
log_density_vec = ravelize_function(log_density, spec_to_pytree(parameter_spec))


# we might also want something like "generated quantities"
@jit
def generated_quantities(rng, x_new, **params):
    constrained, _ = constrain(parameter_spec, **params)
    alpha, beta, sigma = constrained["alpha"], constrained["beta"], constrained["sigma"]
    mu_new = alpha + x_new @ beta
    y_new = mu_new + sigma * random.normal(rng, shape=x_new.shape)
    return {"alpha": alpha, "beta": beta, "sigma": sigma, "y_new": y_new}


# and a flattened version
@jit
def generated_quantities_vec(rng, x_new, params_vec):
    gq = lambda param_dict: generated_quantities(rng, x_new, **param_dict)
    return ravelize_function(gq, spec_to_pytree(parameter_spec))(params_vec)
