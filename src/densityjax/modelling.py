import jax
from typing import Mapping

from .constraints import ParameterConstraint, unconstrain, constrain, spec_to_pytree


# version that assumes data is closed over in
# the passed-in functions.
# Could easily change to pass data later
def make_log_density(
    log_prior,
    log_likelihood,
    parameter_spec: Mapping[str, ParameterConstraint] = dict(),
):
    """
    Make a log_density function from a log_prior, log_likelihood,
    and (optionally) a function to constrain parameters.

    Parameters
    ----------
    log_prior : function
        This function will be passed the parameters
        by name, and should return the log of the prior density.
    log_likelihood : function.
        This function will be passed the parameters
        by name, and should return the log of the likelihood.
    parameter_spec : dict
        A dictionary mapping parameter names to ParameterConstraint
        objects.

    Returns
    -------
    function
        A function that computes the log density of the model.
    """

    @jax.jit
    def log_density(unc_params):
        params, log_det_jac = constrain(parameter_spec, **unc_params)
        return log_det_jac + log_prior(**params) + log_likelihood(**params)

    return log_density


# similar to a solution found at https://github.com/jax-ml/jax/discussions/9508#discussioncomment-2144076,
# but uses ravel_pytree to avoid needing to split the key
def init_random(parameter_spec, rng_key, radius=2):
    """
    Given a tree and a random key, return a tree with the same structure
    but with each leaf replaced by a random uniform value in the range [-radius, radius].
    """
    d, unravel = jax.flatten_util.ravel_pytree(spec_to_pytree(parameter_spec))
    uniforms = jax.random.uniform(rng_key, shape=d.shape, minval=-radius, maxval=radius)
    return unravel(uniforms)


def init(parameter_spec, parameters):
    return unconstrain(parameter_spec, **parameters)
