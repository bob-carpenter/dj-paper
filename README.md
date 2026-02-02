# JAX à la Stan
Brian Ward, Mitzi Morris, Andrew Gelman, Bob Carpenter
2026-01-25

**

[![build and
publish](https://github.com/bob-carpenter/dj-paper/actions/workflows/build.yml/badge.svg)](https://github.com/bob-carpenter/dj-paper/actions/workflows/build.yml)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

### Authors

- [Brian Ward](https://brianward.dev) (Flatiron Institute)
- [Mitzi Morris](https://mitzimorris.github.io) (Independent Contractor)

### Abstract

This paper shows how we can code Bayesian models directly in Python with
JAX by following the pattern developed in the Stan probabilistic
programming language. This allows a direct, line-by-line translation of
all of the courses, texts, and case studies for Stan across the
physical, biological, and social sciences, engineering, business,
health, education, policy, economics, and sports. In the general case, a
Bayesian model is defined by the joint density of observed data and
latent parameters. State-of-the-art algorithms require first or second
derivatives of the log density with respect to the parameters. Stan was
thus designed to make it easy to code differentiable log densities and
generate posterior predictive quantities. Coupled with modern hardware
(e.g., multi-core, GPU, and TPU), compiled JAX far exceeds the
efficiency and scalabilty of Stan. The package ArviZ provides the same
posterior analysis tools as Stan, and Blackjax provides a wider range of
sampling algorithms. Thus we can bring all of the benefits of Stan-style
model building to modern hardware without sacrificing the benefits of a
Python integrated programming environment.
