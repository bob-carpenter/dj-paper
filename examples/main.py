import functools
import blackjax
import jax
import jax.numpy as jnp

from util import init_random

from linear_regression import (
    log_density,
    generated_quantities,
    parameter_spec,
)
from linear_regression import log_density_vec, generated_quantities_vec

from time_series_models import (
    log_density as log_density_garch,
    generated_quantities as generated_quantities_garch,
    parameter_spec as parameter_spec_garch,
    log_density_simplex as log_density_garch_simplex,
    generated_quantities_simplex as generated_quantities_garch_simplex,
    parameter_spec_simplex as parameter_spec_garch_simplex,
    log_density_vec as log_density_vec_garch,
    generated_quantities_vec as generated_quantities_vec_garch,
)


def stan_sample(log_density, initial, steps=1_000, rng_key=None):
    # completely copied from https://blackjax-devs.github.io/blackjax/examples/quickstart.html

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    warmup = blackjax.window_adaptation(blackjax.nuts, log_density)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial, num_steps=steps)

    kernel = blackjax.nuts(log_density, **parameters).step
    states = inference_loop(sample_key, kernel, state, steps)
    return states


if __name__ == "__main__":
    N = 1000
    rng_key = jax.random.key(4567)

    ####################### linear regression ##########################
    init_key, sample_key, gq_key = jax.random.split(rng_key, 3)

    # for "generated quantities"-like behavior:
    rngs = jax.random.split(gq_key, N)
    x_new = jnp.array([0.1, 0.4])

    # sample
    init_draw = init_random(parameter_spec, init_key)
    states = stan_sample(log_density, init_draw, N, sample_key)
    # postprocess draws - constrains and does generated quantities
    draws = jax.vmap(generated_quantities, (0, None))(rngs, x_new, **states.position)

    print("linear regression: ", jax.tree.map(functools.partial(jnp.mean, axis=0), draws))

    # ------------- "flat" version -------------

    init_draw_vec = jax.random.uniform(init_key, shape=(4,))
    states_vec = stan_sample(log_density_vec, init_draw_vec, N, sample_key)
    draws_vec = jax.vmap(generated_quantities_vec, (0, None, 0))(
        rngs, x_new, states_vec.position
    )
    # note: because generated_quantities returns a pytree, we're no longer
    # in the flattened realm
    print("linear regression: ", jax.tree.map(functools.partial(jnp.mean, axis=0), draws_vec))


    ####################### garch (1,1) ##########################
    # reuses the same random keys as the linear regression example
    r_new = jnp.array([0.1])

    # sample
    init_draw_garch = init_random(parameter_spec_garch, init_key)
    states = stan_sample(log_density_garch, init_draw_garch, N, sample_key)
    # postprocess draws - constrains and does generated quantities
    draws = jax.vmap(generated_quantities_garch, (0, None))(rngs, r_new, **states.position)
    
    print("garch (1,1): ", jax.tree.map(functools.partial(jnp.mean, axis=0), draws))

    # ------------- "flat" version -------------
    
    init_draw_vec_garch = jax.random.uniform(init_key, shape=(3,))
    states_vec_garch = stan_sample(log_density_vec_garch, init_draw_vec_garch, N, sample_key)
    draws_vec_garch = jax.vmap(generated_quantities_vec_garch, (0, None, 0))(
        rngs, r_new, states_vec_garch.position
    )
    # note: because generated_quantities returns a pytree, we're no longer
    # in the flattened realm
    print("garch (1,1): ", jax.tree.map(functools.partial(jnp.mean, axis=0), draws_vec_garch))

    # ------------- simplex version (alpha_1, beta_1, slack) -------------

    init_draw_simplex = init_random(parameter_spec_garch_simplex, init_key)
    states_simplex = stan_sample(log_density_garch_simplex, init_draw_simplex, N, sample_key)
    draws_simplex = jax.vmap(generated_quantities_garch_simplex, (0, None))(
        rngs, r_new, **states_simplex.position
    )
    print("garch (1,1) simplex: ", jax.tree.map(functools.partial(jnp.mean, axis=0), draws_simplex))
