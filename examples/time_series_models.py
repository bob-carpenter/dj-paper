from jax import random, jit, lax
import jax
import jax.numpy as jnp

from densityjax import bind, make_log_density, positive, box, simplex, normal
from densityjax.constraints import constrain, spec_to_pytree
from densityjax.util import ravelize_function

# These are the primary exports of this module:
__all__ = [
    "log_density",
    "log_density_vec",
    "constraints",
    "generated_quantities",
    "generated_quantities_vec",
    "parameter_spec_simplex",
    "log_density_simplex",
    "generated_quantities_simplex",
]

# ---------- simulate data from a garch(1,1) ----------
rng_key = random.key(0)
data_key, rng_key = random.split(rng_key)
true = {"alpha_0": 0.1, "alpha_1": 0.1, "beta_1": 0.85}
T = 3000


def simulate(key, alpha_0, alpha_1, beta_1, T):
    def step(s2, e):
        r = jnp.sqrt(s2) * e
        return alpha_0 + alpha_1 * r**2 + beta_1 * s2, (r, s2)

    _, (r, s2) = lax.scan(
        step, alpha_0 / (1 - alpha_1 - beta_1), random.normal(key, (T,))
    )
    return r, s2


returns, true_s2 = simulate(data_key, **true, T=T)

## Parameter definitions

# This can be partial (or even entirely omitted) if you do not require any reshaping utilities
#   (ravelize_function, spec_to_pytree, init_random, etc.),
# otherwise it must cover all parameters to get the shapes/dtypes correct.
parameter_spec = {
    "alpha_0": positive(),
    "alpha_1": box(lower=0.0, upper=1.0),
    "beta_1": bind("alpha_1", lambda a: box(lower=0.0, upper=1 - a)),
}


# log density components


def filter_variance(alpha_0, alpha_1, beta_1):
    """Returns (conditional variances for each observation, variance for the next step)."""

    def step(s2, r_prev):
        return alpha_0 + alpha_1 * r_prev**2 + beta_1 * s2, s2

    s2_next, s2 = jax.lax.scan(step, alpha_0 / (1 - alpha_1 - beta_1), returns)
    return s2, s2_next


def log_likelihood(alpha_0, alpha_1, beta_1):
    target = normal(alpha_0, 0.0, 1.0)
    # flat priors on alpha_1, beta_1

    s2, _ = filter_variance(alpha_0, alpha_1, beta_1)
    target += normal(returns, 0.0, jnp.sqrt(s2))
    return target


def _generated_quantities(r_new, alpha_0, alpha_1, beta_1):
    _, s2_curr = filter_variance(alpha_0, alpha_1, beta_1)
    sigma_new = jnp.sqrt(alpha_0 + alpha_1 * r_new**2 + beta_1 * s2_curr)
    return {
        "alpha_0": alpha_0,
        "alpha_1": alpha_1,
        "beta_1": beta_1,
        "sigma_new": sigma_new,
    }


# a log density function
log_density = make_log_density(log_likelihood, parameter_spec=parameter_spec)


# We can also provide a flattened version, automatically,
# using the structure of the parameters defined above.
log_density_vec = ravelize_function(log_density, spec_to_pytree(parameter_spec))


# we might also want something like "generated quantities"
@jit
def generated_quantities(rng, r_new, **params):
    constrained, _ = constrain(parameter_spec, **params)
    return _generated_quantities(
        r_new, constrained["alpha_0"], constrained["alpha_1"], constrained["beta_1"]
    )


# and a flattened version
@jit
def generated_quantities_vec(rng, r_new, params_vec):
    gq = lambda param_dict: generated_quantities(rng, r_new, **param_dict)
    return ravelize_function(gq, spec_to_pytree(parameter_spec))(params_vec)


# ---------- same model, (alpha_1, beta_1, slack) as a 3-simplex ----------
# Tests the simplex constraint: w = (alpha_1, beta_1, 1 - alpha_1 - beta_1),
# so alpha_1 + beta_1 < 1 holds by construction. The shape is that of the
# constrained parameter, hence (3,) for a 3-simplex.
parameter_spec_simplex = {
    "alpha_0": positive(),
    "w": simplex(shape=(3,)),
}


@jit
def log_density_simplex(unc_params):
    params, log_det_jac = constrain(parameter_spec_simplex, **unc_params)
    alpha_0, alpha_1, beta_1 = params["alpha_0"], params["w"][0], params["w"][1]
    return log_det_jac + log_likelihood(alpha_0, alpha_1, beta_1)


@jit
def generated_quantities_simplex(rng, r_new, **params):
    constrained, _ = constrain(parameter_spec_simplex, **params)
    return _generated_quantities(
        r_new, constrained["alpha_0"], constrained["w"][0], constrained["w"][1]
    )
