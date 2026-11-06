import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
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
    _, unravel = ravel_pytree(pytree)
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


class bind:
    """Constraint built from earlier parameters: bind("a", "b", lambda a, b: box(...))."""
    def __init__(self, *args):
        *parents, self.f = args
        self.parents = tuple(parents)
        self.depends_on = frozenset(self.parents)

    def make(self, theta):
        return self.f(*(theta[p] for p in self.parents))


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
        return jnp.sum(x)


# basic example of a box constraint; either bound may be infinite, per element
# (logistic if both are finite, exp if one is, identity if neither, as in Stan)
class box(ParameterConstraint):
    def __init__(self, lower=-jnp.inf, upper=jnp.inf, **kwargs):
        self.lower = lower
        self.upper = upper
        super().__init__(**kwargs)

    def _finite_bounds(self):
        # jnp.where differentiates every branch, so the unused ones must stay finite:
        # l < u are the bounds where finite and stand-ins otherwise
        has_l, has_u = jnp.isfinite(self.lower), jnp.isfinite(self.upper)
        l = jnp.where(has_l, self.lower, jnp.where(has_u, self.upper - 1.0, 0.0))
        u = jnp.where(has_u, self.upper, l + 1.0)
        return has_l & has_u, has_l & ~has_u, ~has_l & has_u, l, u

    def __call__(self, x):
        both, lower_only, upper_only, l, u = self._finite_bounds()
        return jnp.where(both, l + (u - l) * jax.nn.sigmoid(x),
               jnp.where(lower_only, l + jnp.exp(jnp.where(lower_only, x, 0.0)),
               jnp.where(upper_only, u - jnp.exp(-jnp.where(upper_only, x, 0.0)), x)))

    def inverse(self, y):
        both, lower_only, upper_only, l, u = self._finite_bounds()
        inside = lambda mask: jnp.where(mask, y, (l + u) / 2)  # keeps log arguments positive
        return jnp.where(both, jnp.log(inside(both) - l) - jnp.log(u - inside(both)),
               jnp.where(lower_only, jnp.log(inside(lower_only) - l),
               jnp.where(upper_only, -jnp.log(u - inside(upper_only)), y)))

    def jacobian(self, x):
        both, lower_only, upper_only, l, u = self._finite_bounds()
        return jnp.sum(jnp.where(both, jnp.log(u - l) - jax.nn.softplus(x) - jax.nn.softplus(-x),
                       jnp.where(lower_only, x,
                       jnp.where(upper_only, -x, 0.0))))


# basic example of a sum-to-zero constraint along one or more axes 
# (Householder map, as in PyMC)
class zero_sum(ParameterConstraint):
    def __init__(self, shape, axes=-1, **kwargs):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        axes = (axes,) if isinstance(axes, int) else axes
        self.axes = tuple(sorted({range(len(shape))[a] for a in axes}))  # normalize, dedupe
        self.constrained_shape = shape
        free = tuple(d - (i in self.axes) for i, d in enumerate(shape))
        super().__init__(shape=free, **kwargs)

    def __call__(self, x):
        for a in self.axes:
            r = (x.shape[a] + 1) ** 0.5
            s = x.sum(a, keepdims=True)
            x = jnp.concatenate([x - s / (r * (r + 1)), -s / r], a)
        return x

    def inverse(self, y):
        for a in self.axes:
            r = y.shape[a] ** 0.5
            s = y.sum(a, keepdims=True)
            head, last = jnp.split(y, [y.shape[a] - 1], a)
            y = head - (last + s / r) / (r + 1)
        return y

    def jacobian(self, x):
        return 0.0  # isometry onto the zero-sum subspace


# basic example of a simplex constraint along one axis 
# (softmax of zero-sum logits)
class simplex(zero_sum):
    def __init__(self, shape, axis=-1, **kwargs):
        super().__init__(shape, axes=axis, **kwargs)
        if len(self.axes) != 1:
            raise ValueError("simplex requires exactly one axis.")
        self.axis = self.axes[0]

    def __call__(self, x):
        return jax.nn.softmax(super().__call__(x), axis=self.axis)

    def inverse(self, p):
        return super().inverse(jnp.log(p))

    def jacobian(self, x):
        logp = jax.nn.log_softmax(super().__call__(x), axis=self.axis)
        k = logp.shape[self.axis]
        n_simplexes = logp.size // k
        return logp.sum() + 0.5 * n_simplexes * jnp.log(k)


def spec_to_pytree(
    parameter_spec: Mapping[str, Shaped | bind],
) -> dict[str, jnp.ndarray]:
    """
    Turns a dictionary from parameter names to shapes/dtypes
    into a pytree of zeros with those shapes/dtypes.

    Entries can be anything satisfying the Shaped protocol (ParameterConstraint
    objects, jax.numpy arrays) or a bind, whose shape comes from the
    constraint it builds.
    """
    out = {}
    def get(name):
        if name not in out:
            v = parameter_spec[name]
            if isinstance(v, bind):
                v = v.make({p: get(p) for p in v.parents})
            out[name] = jnp.zeros(v.shape, dtype=v.dtype)
        return out[name]
    return {k: get(k) for k in parameter_spec}


def constrain(parameter_spec: Mapping[str, ParameterConstraint], **kwargs):
    """
    Constrain parameters per the spec and compute the log Jacobian.
    Anything missing from the spec is unconstrained. Binds resolve their
    parents first (at trace time, so no cost under jit).
    """
    parameters = {}
    jacobian = 0.0

    def get(name):
        nonlocal jacobian
        if name not in parameters:
            c = parameter_spec.get(name, real())
            if isinstance(c, bind):
                c = c.make({p: get(p) for p in c.parents})
            parameters[name] = c(kwargs[name])
            jacobian += c.jacobian(kwargs[name])
        return parameters[name]
    
    # Depth-first traversal of the bind dependency graph: get() constrains
    # a bind's parents before the bind itself and skips anything already
    # done, so each parameter and its Jacobian term is computed exactly
    # once, whatever the order of kwargs.
    for name in kwargs:
        get(name)
    return parameters, jacobian


def unconstrain(parameter_spec: Mapping[str, ParameterConstraint], **kwargs):
    """Inverse of constrain. Anything missing from the spec is unconstrained."""
    parameters = {}
    for name, y in kwargs.items():
        c = parameter_spec.get(name, real())
        if isinstance(c, bind):
            c = c.make(kwargs)  # parents are already constrained here
        parameters[name] = c.inverse(y)
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
