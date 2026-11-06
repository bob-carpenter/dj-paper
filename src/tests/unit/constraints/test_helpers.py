"""helpers: constrain, unconstrain and spec_to_pytree on specs with several parameters."""

import jax.numpy as jnp
import numpy as np

from densityjax.constraints import (
    bind,
    box,
    constrain,
    positive,
    spec_to_pytree,
    unconstrain,
    zero_sum,
)

SPEC = {"s": positive(shape=(3,)), "z": zero_sum(4)}
U = {
    "s": jnp.log(jnp.array([2.0, 3.0, 1.0])),
    "z": jnp.array([1.0, 1.0, 1.0]),
    "extra": jnp.array([7.0, 8.0, 9.0]),
}

# child first, so resolving c has to resolve a and b, and b reaches a a second time
CHAIN = {
    "c": bind("a", "b", lambda a, b: box(lower=b, upper=a, shape=(3,))),
    "b": bind("a", lambda a: box(lower=0.0, upper=a, shape=(3,))),
    "a": positive(shape=(3,)),
}
CHAIN_U = {
    "c": jnp.zeros(3),
    "b": jnp.zeros(3),
    "a": jnp.log(jnp.array([1.0, 2.0, 4.0])),
}


def test_constrain__maps_each_spec_parameter_by_its_constraint():
    theta, _ = constrain(SPEC, **U)
    np.testing.assert_allclose(theta["s"], [2.0, 3.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(theta["z"], [0.5, 0.5, 0.5, -1.5], atol=1e-10)


def test_constrain_of_parameter_missing_from_spec__returns_it_unchanged():
    theta, _ = constrain(SPEC, **U)
    np.testing.assert_allclose(theta["extra"], [7.0, 8.0, 9.0], atol=1e-10)


def test_constrain_log_det__is_sum_of_parameter_log_dets():
    _, log_det = constrain(SPEC, **U)
    np.testing.assert_allclose(log_det, np.log(6.0), atol=1e-10)


def test_unconstrain_after_constrain__recovers_every_parameter():
    theta, _ = constrain(SPEC, **U)
    back = unconstrain(SPEC, **theta)
    for k in U:
        np.testing.assert_allclose(back[k], U[k], atol=1e-10, err_msg=k)


def test_constrain_of_chained_binds__builds_each_bound_from_its_constrained_parents():
    theta, _ = constrain(CHAIN, **CHAIN_U)
    np.testing.assert_allclose(theta["a"], [1.0, 2.0, 4.0], atol=1e-10)
    np.testing.assert_allclose(theta["b"], [0.5, 1.0, 2.0], atol=1e-10)
    np.testing.assert_allclose(theta["c"], [0.75, 1.5, 3.0], atol=1e-10)


def test_constrain_log_det_of_chained_binds__is_sum_of_parameter_log_dets():
    _, log_det = constrain(CHAIN, **CHAIN_U)
    np.testing.assert_allclose(log_det, -np.log(64.0), atol=1e-10)


def test_unconstrain_after_constrain_of_chained_binds__recovers_every_parameter():
    theta, _ = constrain(CHAIN, **CHAIN_U)
    back = unconstrain(CHAIN, **theta)
    for k in CHAIN_U:
        np.testing.assert_allclose(back[k], CHAIN_U[k], atol=1e-10, err_msg=k)


def test_spec_to_pytree_of_chained_binds__gives_shape_of_each_built_constraint():
    shapes = {k: v.shape for k, v in spec_to_pytree(CHAIN).items()}
    assert shapes == {"a": (3,), "b": (3,), "c": (3,)}
