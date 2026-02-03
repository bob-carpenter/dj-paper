# JAX à la Stan
Brian Ward, Mitzi Morris, Andrew Gelman, Bob Carpenter
2026-02-02

**

[![build and
publish](https://github.com/bob-carpenter/dj-paper/actions/workflows/build.yml/badge.svg)](https://github.com/bob-carpenter/dj-paper/actions/workflows/build.yml)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

### Authors

- [Brian Ward](https://brianward.dev) (Flatiron Institute)
- [Mitzi Morris](https://mitzimorris.github.io) (Independent Contractor)

### Abstract

We introduce a methodology for coding Bayesian statistical models in
Python with JAX that follows the design pattern of the Stan
probabilistic programming language. This allows a direct, line-by-line
translation into JAX of all of the courses, texts, and case studies for
Stan across the physical, biological, and social sciences, engineering,
business, health, education, policy, economics, and sports, as well as
providing a transparent framework for further model development.
State-of-the-art algorithms require first or second derivatives of an
unnormalized log posterior density. Coupled with modern hardware (e.g.,
multi-core, GPU, and TPU), compiled JAX far exceeds the efficiency and
scalabilty of Stan. JAX’s implementation of NumPy and SciPy, TensorFlow
(including TensorFlow Probability), and DeepMind Distrax provide a wider
range of special function support than Stan. The package ArviZ provides
the same posterior analysis tools as Stan, Blackjax provides a wider
range of inference algorithms, and TensorFlow Probability provides a
wider range of variable transforms. Together, these tools provide an
environment to code models in the style of Stan for modern hardware
without sacrificing the benefits of a Python integrated programming
environment.
