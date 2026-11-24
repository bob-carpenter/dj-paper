"""recurrence and lagged_sum: values against the plain loops they replace."""

import jax.numpy as jnp
import numpy as np
import pytest

from densityjax import lagged_sum, recurrence

# AR(2) with inputs: x[t] = 0.5 x[t-1] - 0.2 x[t-2] + u[t]
AR2 = lambda lags, u_t: 0.5 * lags[0] - 0.2 * lags[1] + u_t
INIT = jnp.array([1.0, 2.0])
U = jnp.array([1.0, -2.0, 0.5, 3.0])


def test_recurrence__matches_python_loop():
    x = [1.0, 2.0]
    for u_t in U:
        x.append(0.5 * x[-1] - 0.2 * x[-2] + u_t)
    np.testing.assert_allclose(recurrence(AR2, INIT, U), x, atol=1e-10)


def test_recurrence_in_reverse__is_forward_recurrence_of_flipped_series():
    backward = recurrence(AR2, INIT, U, reverse=True)
    forward = recurrence(AR2, INIT[::-1], U[::-1])
    np.testing.assert_allclose(backward, forward[::-1], atol=1e-10)


def test_recurrence_with_dict_and_aux__updates_series_together_and_stacks_aux():
    # x[t] = x[t-1] + y[t-1] and y[t] = 2 y[t-1]; the aux result is x[t-1]
    def f(lags):
        x, y = lags["x"][0], lags["y"][0]
        return {"x": x + y, "y": 2 * y}, x

    init = {"x": jnp.array([0.0]), "y": jnp.array([1.0])}
    series, aux = recurrence(f, init, length=3, has_aux=True)
    np.testing.assert_allclose(series["x"], [0.0, 1.0, 3.0, 7.0], atol=1e-10)
    np.testing.assert_allclose(series["y"], [1.0, 2.0, 4.0, 8.0], atol=1e-10)
    np.testing.assert_allclose(aux, [0.0, 1.0, 3.0], atol=1e-10)


def test_recurrence_with_integer_init_and_float_step__raises():
    # rather than rounding every new value to an integer
    with pytest.raises(TypeError):
        recurrence(lambda lags: 1.5 * lags[0], jnp.array([1]), length=3)


@pytest.mark.parametrize("first_lag", [0, 1, 4])
def test_lagged_sum__matches_definition(first_lag):
    # y[t] = sum over j of kernel[j] x[t - first_lag - j], with x = 0 before its start
    kernel, x = jnp.array([0.5, 0.3, 0.2]), jnp.arange(1.0, 7.0)
    y = np.zeros(len(x))
    for t in range(len(x)):
        for j in range(len(kernel)):
            if t - first_lag - j >= 0:
                y[t] += kernel[j] * x[t - first_lag - j]
    np.testing.assert_allclose(lagged_sum(kernel, x, first_lag), y, atol=1e-10)


@pytest.mark.parametrize(
    "kernel_shape, x_shape",
    [
        pytest.param((3,), (2, 6), id="one kernel, two series"),
        pytest.param((2, 3), (6,), id="two kernels, one series"),
        pytest.param((2, 3), (2, 6), id="kernel g for series g"),
    ],
)
def test_lagged_sum_of_batch__is_lagged_sum_of_each_row(kernel_shape, x_shape):
    rng = np.random.default_rng(0)
    kernel, x = rng.normal(size=kernel_shape), rng.normal(size=x_shape)
    rows = zip(np.broadcast_to(kernel, (2, 3)), np.broadcast_to(x, (2, 6)))
    np.testing.assert_allclose(
        lagged_sum(kernel, x), [lagged_sum(k, x_g) for k, x_g in rows], atol=1e-10
    )


def test_lagged_sum_with_fft__matches_direct():
    kernel, x = jnp.exp(-jnp.arange(300.0) / 50), jnp.sin(jnp.arange(500.0))
    np.testing.assert_allclose(
        lagged_sum(kernel, x, method="fft"), lagged_sum(kernel, x), atol=1e-10
    )


def test_lagged_sum_with_negative_first_lag__raises():
    with pytest.raises(ValueError):
        lagged_sum(jnp.ones(2), jnp.ones(3), first_lag=-1)
