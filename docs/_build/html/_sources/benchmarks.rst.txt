Benchmarks
==========

Benchmarks should be read as measurements of rank-one modified workflows, not as
claims that ``cholrot`` replaces LAPACK, MKL, or OpenBLAS Cholesky factorization.
The point is to avoid recomputing or storing objects that are unnecessary for a
rank-one modified product or solve.

Dense rank-one benchmark
------------------------

The dense benchmark compares three routes for computing

.. math::

   w = Dv, \qquad D^T D = R^T R - zz^T.

1. build the modified matrix, recompute Cholesky with NumPy, then multiply;
2. materialize ``D`` with ``cholrot.downdate``, then multiply;
3. compute ``D @ v`` directly with ``cholrot.matvec``.

Run the local benchmark with:

.. code-block:: bash

   python benchmarks/bench_rank1.py --sizes 100 200 400 800 1600 --repeat 5

Save CSV output with:

.. code-block:: bash

   python benchmarks/bench_rank1.py --csv benchmarks/results/local.csv

Report enough environment information to make the numbers interpretable:

* CPU model and core/thread count;
* operating system;
* Python version;
* NumPy version and BLAS/LAPACK backend when known;
* BLAS thread settings, for example ``OMP_NUM_THREADS`` or
  ``OPENBLAS_NUM_THREADS``;
* ``cholrot.backend()`` value;
* matrix size ``n``;
* NumPy recompute + matvec time;
* ``cholrot`` materialize + matvec time;
* ``cholrot`` direct ``matvec`` time;
* speedup relative to NumPy recompute.

Threading
---------

The current ``cholrot`` C++ kernels are single-threaded.  NumPy Cholesky may use
multiple BLAS/LAPACK threads depending on the installed backend.  For this
reason, it is useful to report two benchmark modes:

* single-threaded BLAS, to compare algorithmic work more directly;
* default local BLAS settings, to show real-world behavior on the machine used.

Structured identity benchmark
-----------------------------

The identity benchmark measures the special case

.. math::

   D D^T = I + \alpha zz^T.

In this setting ``cholrot.identity_matvec`` can avoid scanning a dense input
factor because the original Cholesky factor is the identity.

Run it with:

.. code-block:: bash

   python benchmarks/bench_identity.py --sizes 100 200 400 800 1600 3200 6400 --repeat 5

Scope of public benchmark results
---------------------------------

Public benchmark tables should correspond to functions that are part of the
published package API: ``update``, ``downdate``, ``matvec``, ``cholsolve``, and
``identity_matvec``.  Results from separate private experiments should not be
mixed into the public benchmark tables.
