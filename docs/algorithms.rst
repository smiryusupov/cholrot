Algorithms
==========

Rank-one Cholesky modification
------------------------------

Let ``R`` be an upper triangular Cholesky factor of a symmetric positive
definite matrix ``A``:

.. math::

   A = R^T R.

A rank-one update or downdate modifies the matrix as

.. math::

   A_{\mathrm{new}} = A + \alpha z z^T,
   \qquad \alpha \in \{+1, -1\}.

The package computes a triangular factor ``D`` such that

.. math::

   D^T D = R^T R + \alpha z z^T

for the upper convention.  For the lower convention, the analogous relation is

.. math::

   D D^T = L L^T + \alpha z z^T.

Downdates require the modified matrix to remain positive definite.  If the
condition fails, ``cholrot`` raises ``NonPositiveDefiniteError``.

Hyperbolic rotations
--------------------

For downdates, the algorithms use hyperbolic rotations to remove the rank-one
term while preserving triangular structure.  ``cholrot`` exposes three related
recurrences:

``method="hy"``
   A direct hyperbolic-rotation recurrence.

``method="hc"``
   A Chambers-style hyperbolic-cosine recurrence.

``method="algorithm_a"``
   An Algorithm-A recurrence using scalar ``lambda`` updates.

These methods are mathematically equivalent for valid inputs but have different
rounding behavior and implementation structure.

Direct factor-vector product
----------------------------

A common workflow is not to inspect the modified factor itself, but to compute

.. math::

   w = D v.

Calling ``downdate(R, z) @ v`` first materializes the full triangular factor
``D``.  ``cholrot.matvec(R, z, v)`` fuses the recurrence with the product and
returns the same result without storing ``D``.

Rank-one modified solve
-----------------------

``cholrot.cholsolve`` solves

.. math::

   (A + \alpha z z^T) x = b

using triangular solves with the original Cholesky factor and the rank-one
inverse identity.  It does not form the modified matrix or the modified
Cholesky factor.


Identity-structured product
---------------------------

For

.. math::

   D D^T = I + \alpha z z^T,

``cholrot.identity_matvec`` computes ``D @ v`` without scanning a dense base
factor.  This is the structured case where the largest benchmark gains are
expected, because the original Cholesky factor is the identity.
