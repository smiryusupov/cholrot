Manual
======

This page is a compact technical manual for the algorithms exposed by
``cholrot``.  It is intended to be readable as HTML and as a generated PDF.

Scope
-----

``cholrot`` focuses on rank-one Cholesky modifications and operations that can
be performed without materializing unnecessary dense objects.  It is not a
replacement for LAPACK, MKL, or other highly optimized dense Cholesky
factorization libraries.  It is useful when a matrix has already been factored
and is then modified by one outer product.

The supported modification is

.. math::

   A_{new} = A + \alpha z z^T, \qquad \alpha \in \{-1,+1\}.

``alpha = +1`` is an update.  ``alpha = -1`` is a downdate.

Notation and conventions
------------------------

Upper-triangular convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``R`` is upper triangular, ``cholrot`` uses

.. math::

   A = R^T R.

The modified factor ``D`` satisfies

.. math::

   D^T D = R^T R + \alpha z z^T.

Lower-triangular convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``L`` is lower triangular, ``cholrot`` uses

.. math::

   A = L L^T.

The modified factor ``D`` satisfies

.. math::

   D D^T = L L^T + \alpha z z^T.

Positive-definiteness
~~~~~~~~~~~~~~~~~~~~~

Updates with ``alpha = +1`` preserve positive definiteness when ``A`` is
positive definite.  Downdates with ``alpha = -1`` are valid only if

.. math::

   A - z z^T

remains positive definite.  If the downdated matrix is not positive definite,
``cholrot`` raises ``NonPositiveDefiniteError``.

Materialized update/downdate
----------------------------

The functions ``update`` and ``downdate`` return the modified Cholesky factor.
For upper-triangular input:

.. code-block:: python

   D = cholrot.downdate(R, z)

and the result is verified by

.. code-block:: python

   np.allclose(D.T @ D, R.T @ R - np.outer(z, z))

For lower-triangular input:

.. code-block:: python

   D = cholrot.downdate(L, z, lower=True)

and the result is verified by

.. code-block:: python

   np.allclose(D @ D.T, L @ L.T - np.outer(z, z))

The generic dense update/downdate cost is quadratic in the matrix dimension,
whereas recomputing Cholesky from scratch is cubic.

Hyperbolic rotations for downdating
-----------------------------------

Downdating is based on hyperbolic rotations.  The goal is to remove the
rank-one term while preserving triangular structure.  In exact arithmetic,
all exposed downdate methods compute the same mathematical factor, up to the
usual sign convention on Cholesky diagonals.

The implementation exposes three method labels.

``method="hy"``
   Direct hyperbolic-rotation recurrence.  This is the most literal
   representation of the downdate operation.

``method="hc"``
   Chambers-style recurrence.  This formulation is often preferable in
   practice because it organizes the scalar updates differently and is a useful
   reference method for benchmarks.

``method="algorithm_a"``
   Algorithm-A-style recurrence.  It is included for comparison and for
   reproducing the experimental benchmark family.

Direct modified-factor products
-------------------------------

A central feature of ``cholrot`` is the ability to compute

.. math::

   w = D v

without first materializing the full modified factor ``D``.

The explicit route is

.. code-block:: python

   D = cholrot.downdate(R, z)
   w = D @ v

The fused route is

.. code-block:: python

   w = cholrot.matvec(R, z, v, alpha=-1)

The fused route avoids storing ``D``.  For dense generic triangular factors the
operation is still quadratic, but it avoids the extra materialization and can
reduce memory traffic.  For structured cases the saving can be much larger.

Matrix right-hand sides
~~~~~~~~~~~~~~~~~~~~~~~

``matvec`` also accepts a matrix of right-hand sides.  If ``B`` has several
columns, then

.. code-block:: python

   W = cholrot.matvec(R, z, B, alpha=-1)

returns the same result as

.. code-block:: python

   W = D @ B

without explicitly storing ``D``.

Rank-one modified solves
------------------------

``cholsolve`` solves

.. math::

   (A + \alpha z z^T)x = b

where ``A`` is represented by its Cholesky factor.  The function avoids
forming the modified matrix explicitly.

For upper-triangular factors:

.. code-block:: python

   x = cholrot.cholsolve(R, z, b, alpha=-1)

The reference identity tested by the package is

.. code-block:: python

   Anew = R.T @ R - np.outer(z, z)
   np.allclose(x, np.linalg.solve(Anew, b))

Identity-structured product
---------------------------

For the special case

.. math::

   D D^T = I + \alpha z z^T,

``cholrot`` provides ``identity_matvec``:

.. code-block:: python

   w = cholrot.identity_matvec(z, v, alpha=-1, lower=True)

This is the structured case in which the largest speedups are expected.  The
algorithm does not scan a dense triangular input factor because the original
factor is the identity.

Correctness tests
-----------------

The regression suite checks the following identities:

* materialized update/downdate factors reconstruct the modified matrix;
* direct ``matvec`` agrees with explicit ``D @ v`` and ``D @ B``;
* ``cholsolve`` agrees with ``numpy.linalg.solve`` on the modified matrix;
* ``identity_matvec`` agrees with the explicit Cholesky factor in the identity
  structured case;
* invalid downdates raise ``NonPositiveDefiniteError``.

Benchmark interpretation
------------------------

Benchmarks should be interpreted by separating algorithmic complexity from
implementation constants and threading.

Full dense Cholesky recomputation costs cubic time.  Dense rank-one
modification costs quadratic time.  The identity-structured product can be
linear in the vector length.

Highly optimized LAPACK/MKL Cholesky with many threads can be competitive for
small and moderate dense matrices.  The reason to use ``cholrot`` is not that
it replaces MKL, but that it avoids recomputing or materializing objects that
are unnecessary for rank-one modified problems.
