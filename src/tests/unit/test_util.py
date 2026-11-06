import jax.numpy as jnp

from densityjax.util import broadcast_shapes_of

import pytest


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param([1.0, jnp.array([1, 2, 3])], (3,), id="scalar, 1-d array"),
        pytest.param(
            [1.0, jnp.array([[1, 2, 3], [2, 4, 6]])], (2, 3), id="scalar, 2-d array"
        ),
        pytest.param(
            [jnp.array([[2, 1, 0]]), jnp.array([[1, 2, 3], [2, 4, 6]])],
            (2, 3),
            id="both 2-d array",
        ),
    ],
)
def test_broadcast_shapes(args, expected):
    assert broadcast_shapes_of(*args) == expected
