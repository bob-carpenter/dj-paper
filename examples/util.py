import jax
import jax.numpy as jnp
from typing import Mapping, Protocol


def ravelize_function(f, pytree):
    """
    Takes a function that accepts a PyTree and a PyTree,
    and produces a function that accepts a flat array.
    """
    # note: ravel_pytree is only really safe when we
    # know all the dtypes are the same. See
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.flatten_util.ravel_pytree.html
    # This is usually true in stats models
    _, unravel = jax.flatten_util.ravel_pytree(pytree)
    return lambda x: f(unravel(x))


class Shaped(Protocol):
    shape: tuple[int, ...]
    dtype: jnp.dtype


class ParameterConstraint:
    def __init__(self, shape=(), dtype=jnp.float32):
        # if any(s < 0 for s in shape):
        # raise ValueError("Shape dimensions must be non-negative")

        self.shape = shape
        self.dtype = dtype

    def __call__(self, x):
        return x

    def inverse(self, y):
        return y

    def jacobian(self, _):
        return 0.0


# simple alias: base class does transforms
real = ParameterConstraint


# basic example of a positive constraint
class positive(ParameterConstraint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
        return jnp.exp(x)

    def inverse(self, y):
        return jnp.log(y)

    def jacobian(self, x):
        return x


# note: shape will need some care for something like a simplex,
# which should also include axis=... in its definition. The shape
# should be that of the unconstrained parameter.
class simplex(ParameterConstraint):
    def __init__(self, shape, axes=..., **kwargs):
        self.axes = axes
        shape_n = jnp.array(shape, dtype=int)
        shape_n = shape_n.at[axes,].set(shape_n[axes,] - 1)

        super().__init__(shape=tuple(shape_n), **kwargs)

    def __call__(self, x):
        raise NotImplementedError("Simplex constraint not yet implemented")

    def inverse(self, y):
        raise NotImplementedError("Simplex constraint not yet implemented")

    def jacobian(self, x):
        raise NotImplementedError("Simplex constraint not yet implemented")


def spec_to_pytree(
    parameter_spec: Mapping[str, Shaped],
) -> dict[str, jnp.ndarray]:
    """
    Turns a dictionary from parameter names to shapes/dtypes
    into a pytree of zeros with those shapes/dtypes.

    Note that the Shaped protocol is satisfied by ParameterConstraint
    objects, but also by something like a jax.numpy array.
    """
    return {k: jnp.zeros(v.shape, dtype=v.dtype) for k, v in parameter_spec.items()}


def constrain(parameter_spec: Mapping[str, ParameterConstraint], **kwargs):
    """
    Constrain parameters according to the provided spec and compute the log jacobian.
    Anything missing from the spec is assumed to have no constraints.
    """
    jacobian = 0.0
    parameters = {}
    for param in kwargs:
        if param in parameter_spec:
            parameters[param] = parameter_spec[param](kwargs[param])
            jacobian += parameter_spec[param].jacobian(kwargs[param])
        else:
            parameters[param] = kwargs[param]
    return parameters, jacobian


def unconstrain(parameter_spec: Mapping[str, ParameterConstraint], **kwargs):
    """
    Inverse of constrain: given constrained parameters, return unconstrained versions.
    Anything missing from the spec is assumed to have no constraints.
    """
    parameters = {}
    for param in kwargs:
        if param in parameter_spec:
            parameters[param] = parameter_spec[param].inverse(kwargs[param])
        else:
            parameters[param] = kwargs[param]
    return parameters


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
