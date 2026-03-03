# JAX à la Stan
Brian Ward, Matthijs Vákár, Mitzi Morris, Andrew Gelman, Bob Carpenter
2026-02-18

### Authors

- [Brian Ward](https://brianward.dev) (Flatiron Institute)
- [Matthijs Vákár](https://www.uu.nl/staff/MILVakar) (Utrecht
  University)
- [Mitzi Morris](https://mitzimorris.github.io) (Independent Contractor)
- [Andrew Gelman](https://sites.stat.columbia.edu/gelman/) (Columbia
  Univesity)
- [Bob Carpenter](https://brianward.dev) (Flatiron Institute)

[![build and
publish](https://github.com/bob-carpenter/dj-paper/actions/workflows/build.yml/badge.svg)](https://github.com/bob-carpenter/dj-paper/actions/workflows/build.yml)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

### Abstract

We introduce a methodology for coding Bayesian statistical models in
Python with JAX that follows the design pattern of the Stan
probabilistic programming language. This allows a direct, line-by-line
translation into JAX of all of the courses, textbooks, and case studies
for Stan across the physical, biological, and social sciences,
engineering, business, health, education, policy, economics, and sports.
It also provides a transparent framework for further model development.
Coupled with modern hardware (e.g., multi-core, graphics processing
units, and tensor processing units), compiled JAX far exceeds the
efficiency and scalabilty of Stan for computing the log densities and
gradients needed by state-of-the-art inference algorithms. JAX’s
implementation of NumPy and SciPy, along with the packages TensorFlow
(including TensorFlow Probability) and Distrax, provide a much wider
range of special function support than Stan, including partial and
stochastic differential equations and neural networks. The package ArviZ
provides the same posterior analysis tools as Stan, Blackjax provides a
wider range of inference algorithms, and TensorFlow Probability provides
a wider range of variable transforms. Together, these tools provide an
environment to code models in the style of Stan targeting modern
hardware without leaving an integrated Python programming environment.

### Rendered draft

- GitHub pages: [Jax à la
  Stan](https://bob-carpenter.github.io/dj-paper/)
