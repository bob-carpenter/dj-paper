import jax

# finite-difference checks need double precision
jax.config.update("jax_enable_x64", True)
