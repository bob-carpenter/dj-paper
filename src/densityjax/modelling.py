import jax
from typing import Mapping

from .constraints import ParameterConstraint, unconstrain, constrain, spec_to_pytree


def make_log_density(
    log_posterior,
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
        return log_det_jac + log_posterior(**params)

    return log_density


# similar to a solution found at https://github.com/jax-ml/jax/discussions/9508#discussioncomment-2144076,
# but uses ravel_pytree to avoid needing to split the key
def init_random(parameter_spec, rng_key, radius=1.0):
    """
    Given a tree and a random key, return a tree with the same structure
    but with each leaf replaced by a random uniform value in the range [-radius, radius].
    """
    d, unravel = jax.flatten_util.ravel_pytree(spec_to_pytree(parameter_spec))
    uniforms = jax.random.uniform(rng_key, shape=d.shape, minval=-radius, maxval=radius)
    return unravel(uniforms)


def init(parameter_spec, parameters):
    return unconstrain(parameter_spec, **parameters)


class Model:
    def __init__(
        self, parameter_spec, log_posterior, forward=lambda rng, **kwargs: kwargs
    ):
        self._parameter_spec = parameter_spec
        self._log_posterior = log_posterior
        self._forward = forward

    def log_density(self, params):
        constrained_params, log_det_jac = self.constrain(params)
        return log_det_jac + self._log_posterior(**constrained_params)

    def constrain(self, params):
        return constrain(self._parameter_spec, **params)

    def initialize(self, params):
        return unconstrain(self._parameter_spec, **params)

    def initialize_random(self, rng_key, radius=1.0):
        return init_random(self._parameter_spec, rng_key, radius)

    def parameter_spec(self):
        return self._parameter_spec

    def write(self, rng, params, **extras):
        constrained_params, _ = self.constrain(params)
        return self._forward(rng, **constrained_params, **extras)
