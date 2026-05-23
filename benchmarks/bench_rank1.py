# SPDX-License-Identifier: Apache-2.0
"""Local benchmark for rank-one Cholesky downdate products.

Examples
--------

Run from the repository root after installing the package:

    python benchmarks/bench_rank1.py --sizes 100 200 400 800 1600 --repeat 5
    python benchmarks/bench_rank1.py --csv benchmarks/results/local.csv

The benchmark is intentionally dependency-light.  It is not a substitute for a
full ASV suite, but it gives a reproducible starting point for README numbers.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np

from cholrot import backend, downdate, matvec


def timed(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def one_case(n: int, *, seed: int = 0, repeat: int = 5) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, n))
    A = G.T @ G + n * np.eye(n)
    R = np.linalg.cholesky(A).T
    z = 0.05 * rng.normal(size=n)
    v = rng.normal(size=n)

    def numpy_recompute_then_multiply():
        D = np.linalg.cholesky(A - np.outer(z, z)).T
        return D @ v

    def materialize_then_multiply():
        D = downdate(R, z, method="hy")
        return D @ v

    def direct_factor_matvec():
        return matvec(R, z, v, method="hy")

    # Correctness before timing.
    ref = numpy_recompute_then_multiply()
    np.testing.assert_allclose(materialize_then_multiply(), ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(direct_factor_matvec(), ref, rtol=1e-10, atol=1e-10)

    numpy_time = timed(numpy_recompute_then_multiply, repeat)
    materialize_time = timed(materialize_then_multiply, repeat)
    direct_time = timed(direct_factor_matvec, repeat)
    return {
        "n": n,
        "backend": backend(),
        "repeat": repeat,
        "numpy_cholesky_then_matvec_s": numpy_time,
        "cholrot_downdate_then_matvec_s": materialize_time,
        "cholrot_direct_matvec_s": direct_time,
        "direct_speedup_vs_numpy": numpy_time / direct_time if direct_time else float("inf"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"cholrot backend={backend()}")
    print(f"python={sys.version.split()[0]} numpy={np.__version__} platform={platform.platform()}")

    rows = [one_case(n, seed=n, repeat=args.repeat) for n in args.sizes]
    for row in rows:
        print(
            f"n={row['n']:5d}  "
            f"numpy={row['numpy_cholesky_then_matvec_s']:.6f}s  "
            f"materialize={row['cholrot_downdate_then_matvec_s']:.6f}s  "
            f"direct={row['cholrot_direct_matvec_s']:.6f}s  "
            f"speedup={row['direct_speedup_vs_numpy']:.2f}x"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
