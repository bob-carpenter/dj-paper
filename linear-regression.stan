data {
  int<lower=0> N, N_new, P;
  matrix[N, P] x;
  vector[N] y;
  matrix[N_new, P] x_new;
}
parameters {
  real alpha;
  vector[P] beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 2.5);
  sigma ~ exponential(0.5);
  y ~ normal(alpha + x * beta, sigma);
}
generated quantities {
  array[N_new] real y_new = normal_rng(alpha + x_new * beta, sigma);
}
