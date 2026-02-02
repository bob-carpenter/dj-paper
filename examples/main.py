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

    init_key, sample_key, gq_key = jax.random.split(rng_key, 3)

    # for "generated quantities"-like behavior:
    rngs = jax.random.split(gq_key, N)
    x_new = jnp.array([0.1, 0.4])

    # sample
    init_draw = init_random(parameter_spec, init_key)
    states = stan_sample(log_density, init_draw, N, sample_key)
    # postprocess draws - constrains and does generated quantities
    draws = jax.vmap(generated_quantities, (0, None))(rngs, x_new, **states.position)

    print(jax.tree.map(functools.partial(jnp.mean, axis=0), draws))

    # ------------- "flat" version -------------

    init_draw_vec = jax.random.uniform(init_key, shape=(4,))
    states_vec = stan_sample(log_density_vec, init_draw_vec, N, sample_key)
    draws_vec = jax.vmap(generated_quantities_vec, (0, None, 0))(
        rngs, x_new, states_vec.position
    )
    # note: because generated_quantities returns a pytree, we're no longer
    # in the flattened realm
    print(jax.tree.map(functools.partial(jnp.mean, axis=0), draws_vec))
