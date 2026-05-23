References
==========

``cholrot`` implements standard rank-one Cholesky update and downdate
algorithms and exposes them through a compact NumPy-style API.  The package does
not claim novelty for the classical rotation methods themselves.  Its practical
focus is on making these routines easy to install, test, benchmark, and use from
Python, with direct APIs for modified-factor products and rank-one modified
solves.

Background references
---------------------

* LINPACK Cholesky update and downdate routines, including ``DCHUD`` and
  ``DCHDD``.
* Seeger, M. *Low Rank Updates for the Cholesky Decomposition*.
* Chambers-style hyperbolic rotations for Cholesky downdating.

Ecosystem comparisons
---------------------

The closest public interfaces are classical ``cholupdate``-style routines.  The
main difference in ``cholrot`` is that the package also exposes product and solve
operations that do not require materializing the modified factor.

Useful comparison points include:

* MATLAB ``cholupdate``;
* JAX ``jax.lax.linalg.cholesky_update``;
* TensorFlow Probability ``tfp.math.cholesky_update``.
