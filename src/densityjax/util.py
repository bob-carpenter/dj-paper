from jax.flatten_util import ravel_pytree


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
