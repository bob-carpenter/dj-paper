"""Contract of the constraints family: every invariant holds for every entry in CASES."""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from util import box, positive, simplex, zero_sum

class Case(NamedTuple):
    make: Callable                   # () -> constraint, 3 to 5 elements per axis
    in_support: Callable             # (c, y) -> True iff y is finite and satisfies the constraint
    free: Callable = lambda c, y: y  # (c, y) -> y without outputs determined by the others

# helpers
def fd_jacobian(f, x, eps=1e-6):
    """Central-difference Jacobian of f at x, shape (f(x).size, x.size)."""
    x = np.asarray(x, dtype=np.float64)

    def at(dx):
        return np.asarray(f(jnp.asarray((x.ravel() + dx).reshape(x.shape)))).ravel()

    return np.stack([(at(e) - at(-e)) / (2 * eps) for e in np.eye(x.size) * eps], axis=1)


def fd_log_volume(f, x, eps=1e-6):
    """log of the factor by which f scales volume at x: 0.5 * log det(J^T J).

    Equals log|det J| for square J; for tall J it is the volume factor onto
    the image subspace.
    """
    J = fd_jacobian(f, x, eps)
    return 0.5 * np.linalg.slogdet(J.T @ J)[1]


def is_positive(c, y):
    return np.isfinite(y).all() and (y > 0).all()


def is_box(c, y):
    return np.isfinite(y).all() and (y > c.lower).all() and (y < c.upper).all()


def is_zero_sum(c, y):
    sums = [y.sum(axis=a) for a in c.axes]
    return np.isfinite(y).all() and all(np.allclose(s, 0.0, atol=1e-10) for s in sums)


def is_simplex(c, y):
    sums = y.sum(axis=c.axis)
    return np.isfinite(y).all() and (y > 0).all() and np.allclose(sums, 1.0, atol=1e-10)


def drop_last(c, y):
    """y without its last entry along c.axis, which is 1 minus the sum of the others."""
    return np.delete(y, -1, axis=c.axis)


CASES = {
    "positive":         Case(make=lambda: positive(shape=(3,)), in_support=is_positive),
    "box":              Case(make=lambda: box(lower=-1.0, upper=3.0, shape=(3,)), in_support=is_box),
    "box_array_bounds": Case(make=lambda: box(lower=jnp.array([-1.0, 0.0, 2.0]),
                                              upper=jnp.array([1.0, 4.0, 3.0]), shape=(3,)),
                             in_support=is_box),
    "box_lower_only":   Case(make=lambda: box(lower=2.0, shape=(3,)), in_support=is_box),
    "box_upper_only":   Case(make=lambda: box(upper=-1.0, shape=(3,)), in_support=is_box),
    "box_mixed_bounds": Case(make=lambda: box(lower=jnp.array([-1.0, 2.0, -jnp.inf, -jnp.inf]),
                                              upper=jnp.array([3.0, jnp.inf, 1.0, jnp.inf]), shape=(4,)),
                             in_support=is_box),
    "zero_sum_1d":      Case(make=lambda: zero_sum(5), in_support=is_zero_sum),
    "zero_sum_2d":      Case(make=lambda: zero_sum((3, 4), axes=(0, 1)), in_support=is_zero_sum),
    "zero_sum_3d":      Case(make=lambda: zero_sum((3, 4, 5), axes=(0, 1, 2)), in_support=is_zero_sum),
    "zero_sum_4d":      Case(make=lambda: zero_sum((3, 4, 3, 3), axes=(0, 1, 2, 3)), in_support=is_zero_sum),
    "simplex_1d":       Case(make=lambda: simplex(4), in_support=is_simplex, free=drop_last),
    "simplex_2d":       Case(make=lambda: simplex((3, 4)), in_support=is_simplex, free=drop_last),
    "simplex_2d_axis0": Case(make=lambda: simplex((3, 4), axis=0), in_support=is_simplex, free=drop_last),
}

OUTPUTS = {"forward": lambda c, x: c(x), "log_det": lambda c, x: c.jacobian(x)}

each_case = pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
each_output = pytest.mark.parametrize("output", list(OUTPUTS.values()), ids=list(OUTPUTS))


def random_x(c, batch=()):
    """Fixed-seed standard normal draw of the unconstrained shape, optionally batched."""
    return jnp.asarray(np.random.default_rng(0).normal(size=batch + c.shape))


@each_case
def test_inverse_after_forward__recovers_x(case):
    c = case.make()
    x = random_x(c)
    np.testing.assert_allclose(c.inverse(c(x)), x, atol=1e-10)


@each_case
def test_forward_output__lies_in_support(case):
    c = case.make()
    assert case.in_support(c, np.asarray(c(random_x(c))))


@each_case
def test_log_det__matches_finite_differences(case):
    c = case.make()
    x = random_x(c)
    expected = fd_log_volume(lambda u: case.free(c, c(u)), x)
    np.testing.assert_allclose(c.jacobian(x), expected, atol=1e-6)


@each_output
@each_case
def test_output_under_jit__equals_eager_output(case, output):
    c = case.make()
    x = random_x(c)
    np.testing.assert_allclose(jax.jit(lambda u: output(c, u))(x), output(c, x), atol=1e-10)


@each_output
@each_case
def test_output_under_vmap__equals_output_per_item(case, output):
    c = case.make()
    xs = random_x(c, batch=(3,))
    per_item = np.stack([output(c, x) for x in xs])
    np.testing.assert_allclose(jax.vmap(lambda u: output(c, u))(xs), per_item, atol=1e-10)


""" tests helpers for finite-difference checks (only for debugging)
# R^3 -> R^4 with J = TALL everywhere; J^T J = I + 1 1^T has eigenvalues 4, 1, 1
TALL = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])


def test_fd_jacobian_of_linear_map__is_its_matrix():
    J = fd_jacobian(lambda x: jnp.asarray(TALL) @ x, jnp.ones(3))
    np.testing.assert_allclose(J, TALL, atol=1e-6)


def test_fd_jacobian_of_elementwise_exp__is_diagonal_of_exp_x():
    # x = (0, log 2, log 3): d exp(x_i) / d x_j = exp(x_i) if i == j, else 0
    x = jnp.log(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(fd_jacobian(jnp.exp, x), np.diag([1.0, 2.0, 3.0]), atol=1e-6)


def test_fd_log_volume_of_square_map__is_log_abs_det():
    # J = diag(-1, 2, 3): |det J| = 6; the negative entry checks the absolute value
    D = jnp.diag(jnp.array([-1.0, 2.0, 3.0]))
    np.testing.assert_allclose(fd_log_volume(lambda x: D @ x, jnp.ones(3)), np.log(6.0), atol=1e-6)


def test_fd_log_volume_of_tall_map__is_half_log_det_of_gram():
    # det(J^T J) = det(I + 1 1^T) = 4 * 1 * 1, so 0.5 * log 4 = log 2
    f = lambda x: jnp.asarray(TALL) @ x
    np.testing.assert_allclose(fd_log_volume(f, jnp.ones(3)), np.log(2.0), atol=1e-6)
"""