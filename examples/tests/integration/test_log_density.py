"""Workflow: spec + prior + likelihood -> make_log_density -> evaluate on unconstrained values."""
import jax
import jax.numpy as jnp
import numpy as np

from util import bind, box, make_log_density, positive


def test_log_density_with_exponential_prior__integrates_to_one():
    # s ~ Exponential(1) with s = exp(x): log_density(x) = x - exp(x), whose integral
    # over R is 1 (substitute s = exp(x)). Outside [-40, 40] the integrand is < 1e-17.
    log_density = make_log_density(lambda s: -s, lambda s: 0.0, {"s": positive()})
    x = jnp.linspace(-40.0, 40.0, 20_001)
    q = jnp.exp(jax.vmap(lambda xi: log_density({"s": xi}))(x))
    np.testing.assert_allclose(jnp.sum(q) * (x[1] - x[0]), 1.0, rtol=1e-6)


def test_log_density_with_uniform_prior_bounded_by_bind__integrates_to_one():
    # s ~ Exponential(1) with s = exp(x_s); t | s ~ Uniform(0, s) with t = s sigmoid(x_t), the
    # box bound to s through bind. With log-Jacobians x_s and log s + log sigmoid'(x_t):
    #   log_density = (-s + x_s) + (-log s) + (log s + log sigmoid'(x_t))
    #               = (x_s - exp(x_s)) + log sigmoid'(x_t),
    # a product of two densities that each integrate to 1 over R. log s cancels only if the
    # box is built from the constrained s. Outside [-40, 40] each factor is < 1e-17; the
    # integrand is smooth, so a spacing of 0.1 is enough.
    spec = {"s": positive(), "t": bind("s", lambda s: box(lower=0.0, upper=s))}
    log_density = make_log_density(lambda s, t: -s - jnp.log(s), lambda s, t: 0.0, spec)
    x = jnp.linspace(-40.0, 40.0, 801)
    q = jnp.exp(jax.vmap(lambda xs: jax.vmap(lambda xt: log_density({"s": xs, "t": xt}))(x))(x))
    np.testing.assert_allclose(jnp.sum(q) * (x[1] - x[0]) ** 2, 1.0, rtol=1e-6)


def test_log_density_with_exponential_prior_bounded_below_by_bind__integrates_to_one():
    # s ~ Exponential(1) with s = exp(x_s); t - s | s ~ Exponential(1) with t = s + exp(x_t), the
    # box bound below by s through bind and unbounded above. With log-Jacobians x_s and x_t:
    #   log_density = (-s + x_s) + (-(t - s) + x_t)
    #               = (x_s - exp(x_s)) + (x_t - exp(x_t)),
    # a product of two densities that each integrate to 1 over R. Outside [-40, 40] each factor
    # is < 1e-17; the integrand is smooth, so a spacing of 0.1 is enough.
    spec = {"s": positive(), "t": bind("s", lambda s: box(lower=s))}
    log_density = make_log_density(lambda s, t: -s - (t - s), lambda s, t: 0.0, spec)
    x = jnp.linspace(-40.0, 40.0, 801)
    q = jnp.exp(jax.vmap(lambda xs: jax.vmap(lambda xt: log_density({"s": xs, "t": xt}))(x))(x))
    np.testing.assert_allclose(jnp.sum(q) * (x[1] - x[0]) ** 2, 1.0, rtol=1e-6)