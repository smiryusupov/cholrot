cholrot
=======

``cholrot`` provides rank-one Cholesky update/downdate routines, direct
modified-factor products, and rank-one modified Cholesky solves.

The project is deliberately focused on the hyperbolic-rotation rank-one case:

.. math::

   A_{\mathrm{new}} = A + \alpha z z^T, \qquad \alpha \in \{-1, +1\}.

If ``R`` is an upper Cholesky factor, ``A = R^T R``.  If ``L`` is a lower
Cholesky factor, ``A = L L^T``.

The key public API is:

* ``update`` and ``downdate`` to materialize the modified Cholesky factor;
* ``matvec`` to compute ``D @ v`` without materializing ``D``;
* ``cholsolve`` to solve ``(A + alpha z z.T) x = b`` without forming the
  modified matrix.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   manual
   algorithms
   api
   benchmarks
   references
   license
   release
