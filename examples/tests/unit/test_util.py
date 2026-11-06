"""util (non-family part): constrain, unconstrain and spec_to_pytree on specs with several parameters.

SPEC: s is positive (3 entries), z is zero_sum(4); 'extra' is not in SPEC, so it is real.
Hand values:
  s:     exp(log 2, log 3, log 1) = (2, 3, 1), log-det = log 2 + log 3 + 0 = log 6
  z:     zero_sum(4)(1, 1, 1) = (1, 1, 1, -3)/2, log-det = 0  (unit/constraints/test_values.py)
  extra: unchanged, log-det = 0
  total log-det = log 6

CHAIN: a is positive, b = bind(a -> box(0, a)), c = bind(a, b -> box(b, a)), so c is bound to a
parameter and to a bind. At x = 0, box(l, u) gives l + (u - l) sigmoid(0) = (l + u)/2, with
log-det = sum(log(u - l)) - 3 * 2 softplus(0) = sum(log(u - l)) - log 64.
Hand values:
  a:     exp(log 1, log 2, log 4) = (1, 2, 4), log-det = 0 + log 2 + log 4 = log 8
  b:     (0 + (1, 2, 4))/2 = (1/2, 1, 2), log-det = log(1 * 2 * 4) - log 64 = log 8 - log 64
  c:     ((1/2, 1, 2) + (1, 2, 4))/2 = (3/4, 3/2, 3), log-det = log(1/2 * 1 * 2) - log 64 = -log 64
  total log-det = log 8 + (log 8 - log 64) - log 64 = -log 64
"""
import jax.numpy as jnp
import numpy as np

from util import bind, box, constrain, positive, spec_to_pytree, unconstrain, zero_sum

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
CHAIN_U = {"c": jnp.zeros(3), "b": jnp.zeros(3), "a": jnp.log(jnp.array([1.0, 2.0, 4.0]))}


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
