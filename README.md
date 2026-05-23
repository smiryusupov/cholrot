# cholrot

`cholrot` is a small experimental NumPy package for **rank-one Cholesky
updates/downdates**, **implicit modified-factor products**, and **rank-one
modified Cholesky solves**.

The package is intentionally narrower than MATLAB-style `cholupdate`: besides
returning the modified Cholesky factor, it exposes operations that avoid forming
intermediate matrices when you only need a product or a solve.

## Why this exists

Suppose `R` is a Cholesky factor of a positive definite matrix `A`:

- upper convention: `A = R.T @ R`, or
- lower convention: `A = L @ L.T`.

After a rank-one modification

```text
A_new = A + alpha * z z.T,       alpha in {+1, -1}
```

it is often wasteful to build `A_new`, recompute its Cholesky factor, and then
multiply or solve. `cholrot` provides:

- `update(...)`: materialize the modified Cholesky factor `D`.
- `downdate(...)`: convenience wrapper for `alpha=-1`.
- `matvec(...)`: compute `D @ v` directly, without materializing `D`.
- `cholsolve(...)`: solve `(A + alpha*z*z.T) x = b` without materializing
  `A_new` or `D`.
- `logdet_rank1(...)`: compute the modified log determinant.

The public API is Python, but the package is designed around compiled kernels.
When the pybind11 extension is available, `update(...)`, `matvec(...)`, and `cholsolve(...)` dispatch
to C++ implementations; otherwise they fall back to the reference NumPy/Python
implementation. `cholsolve(...)` uses triangular solves and the rank-one inverse
identity, so it avoids forming both the modified matrix and the modified factor.


## Binary wheels

Release wheels are built with GitHub Actions and `cibuildwheel` for:

- CPython 3.12, 3.13, and 3.14;
- Linux `manylinux2014` x86_64;
- macOS Intel and Apple Silicon;
- Windows 64-bit.

For most users this means a normal install should use a prebuilt wheel:

```bash
python -m pip install cholrot
```

The source distribution remains available for platforms not covered by the
release matrix, provided a C++17 compiler and pybind11-compatible build
environment are present.

## Compile from a local checkout

The pybind11 extension is compiled automatically when you install from the
repository. You need a C++17 compiler: `g++`/`clang++` on Linux, Xcode Command
Line Tools on macOS, or Microsoft C++ Build Tools on Windows.

```bash
python -m pip install --upgrade pip
python -m pip install -e .[test]
pytest
python -c "import cholrot; print(cholrot.backend())"
```

A successful compiled install prints:

```text
cpp
```

To compare against the reference NumPy/Python backend:

```bash
CHOLROT_PURE_PYTHON=1 pytest
```

Build release artifacts with:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```


## C++ / pybind11 backend

The C++ backend lives in `src/cholrot/_core_ext.cpp` and is built with pybind11.
It currently mirrors the tested Python algorithms for:

- materialized rank-one update/downdate via `update(...)`;
- direct modified-factor products via `matvec(...)`;
- rank-one modified solves via `cholsolve(...)`;
- upper and lower Cholesky conventions;
- `method="hy"`, `method="hc"`, and `method="algorithm_a"`.

You can check the active backend at runtime:

```python
import cholrot
print(cholrot.backend())  # "cpp" or "python"
```

For debugging or correctness comparisons, force the reference backend with:

```bash
CHOLROT_PURE_PYTHON=1 pytest
```

## Minimal example

```python
import numpy as np
from cholrot import downdate, matvec, cholsolve, identity_matvec

rng = np.random.default_rng(0)
n = 6
A = rng.normal(size=(n, n))
A = A.T @ A + n * np.eye(n)
R = np.linalg.cholesky(A).T          # upper Cholesky factor: A = R.T @ R
z = 0.1 * rng.normal(size=n)
v = rng.normal(size=n)

D = downdate(R, z, method="hy")      # D.T @ D = R.T @ R - z z.T
w = matvec(R, z, v, method="hy")     # same as D @ v, but D is not formed
x = cholsolve(R, z, v, alpha=-1)     # solve (A - z z.T) x = v

np.testing.assert_allclose(w, D @ v)
np.testing.assert_allclose(
    x,
    np.linalg.solve(A - np.outer(z, z), v),
)
```

## API conventions

By default, `cholrot` uses the **upper** Cholesky convention:

```text
A = R.T @ R
```

Set `lower=True` for the lower convention:

```text
A = L @ L.T
```

Supported methods:

- `method="hy"`: hyperbolic-rotation form.
- `method="hc"`: Chambers-style hyperbolic-cosine variant.
- `method="algorithm_a"`: Algorithm A variant.


## Non-regression tests

The test suite has two layers:

1. property-style correctness tests comparing `update`, `downdate`, `matvec`,
   `cholsolve`, and `logdet_rank1` against NumPy references;
2. golden-value regression tests in `tests/test_regression.py` to catch silent
   changes in sign conventions, lower/upper orientation, or recurrence details.

Recommended release checks:

```bash
pytest
CHOLROT_PURE_PYTHON=1 pytest
```

The GitHub Actions workflow runs both the compiled/default backend and the pure
Python backend on Linux, macOS, and Windows.

## Benchmarks

Run local dense rank-one benchmarks with:

```bash
python benchmarks/bench_rank1.py --sizes 100 200 400 800 1600 --repeat 5
python benchmarks/bench_rank1.py --csv benchmarks/results/local.csv
```

This benchmark compares three routes for computing `w = D @ v`:

1. recompute Cholesky with NumPy, then multiply;
2. materialize the modified factor with `cholrot.downdate`, then multiply;
3. compute `D @ v` directly with `cholrot.matvec`.

For the identity-structured case,

```text
D D.T = I + alpha * z z.T,
```

run:

```bash
python benchmarks/bench_identity.py --sizes 100 200 400 800 1600 3200 6400 --repeat 5
```

Benchmark tables should report the CPU, OS, Python version, NumPy version,
thread settings, `cholrot.backend()`, matrix size, and the three timings above.
The current C++ backend is single-threaded; NumPy may use a multithreaded BLAS or
LAPACK backend. For this reason, both single-threaded BLAS and default local
thread settings are useful benchmark modes.

The intended benchmark conclusion is not "`cholrot` always beats MKL". The
claim is narrower: for rank-one modified workflows, `cholrot` can avoid full
recomputation and can compute products or solves without materializing the
modified factor.

## Documentation

The repo includes a Sphinx documentation skeleton in `docs/`, suitable for
Read the Docs. Build it locally with:

```bash
python -m pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

Keep the package README short and practical. Put the algorithm derivations, math,
API reference, and benchmark methodology in the Sphinx docs.

## Development status

This is an alpha-stage scientific package. The first public release focuses on
rank-one Cholesky update/downdate routines, direct modified-factor products, and
rank-one modified solves through a tested C++/pybind11 backend with a Python
fallback.

The package implements known numerical linear algebra algorithms. The public
contribution is the packaging, API design, correctness tests, benchmarks, and the
focus on product/solve routines that avoid unnecessary materialization.


## License

`cholrot` is licensed under the Apache License, Version 2.0. See `LICENSE`
and `NOTICE`.

The software is provided on an "AS IS" basis, without warranties or conditions
of any kind. Users are responsible for validating numerical behavior in their
own applications.

Contributions are accepted under the same Apache-2.0 license; see
`CONTRIBUTING.md`. Maintainer release steps are documented in `RELEASE.md`.

## References

Useful background and comparison points include:

- LINPACK Cholesky update/downdate routines, including `DCHUD` and `DCHDD`.
- Seeger, M. *Low Rank Updates for the Cholesky Decomposition*.
- MATLAB `cholupdate`.
- JAX `jax.lax.linalg.cholesky_update`.
- TensorFlow Probability `tfp.math.cholesky_update`.

