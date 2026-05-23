# SPDX-License-Identifier: Apache-2.0
"""Benchmark the structured identity-factor product.

This reproduces the benchmark family in the hyperbolic-rotations note where
D D.T = I - z z.T and we compute w = D @ v without materializing D.
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np

from cholrot import backend, identity_matvec, matvec


def timed(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def one_case(n: int, *, seed: int = 0, repeat: int = 5, lower: bool = True) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    z *= 0.5 / np.linalg.norm(z)  # keep I - z z.T safely SPD
    v = rng.normal(size=n)
    R = np.eye(n)

    def numpy_recompute_then_multiply():
        L = np.linalg.cholesky(np.eye(n) - np.outer(z, z))
        D = L if lower else L.T
        return D @ v

    def generic_rotations():
        return matvec(R, z, v, alpha=-1, lower=lower, method="hy")

    def identity_kernel():
        return identity_matvec(z, v, alpha=-1, lower=lower)

    ref = numpy_recompute_then_multiply()
    np.testing.assert_allclose(generic_rotations(), ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(identity_kernel(), ref, rtol=1e-10, atol=1e-10)

    numpy_time = timed(numpy_recompute_then_multiply, repeat)
    generic_time = timed(generic_rotations, repeat)
    identity_time = timed(identity_kernel, repeat)
    return {
        "n": n,
        "backend": backend(),
        "repeat": repeat,
        "numpy_cholesky_then_matvec_s": numpy_time,
        "cholrot_generic_matvec_s": generic_time,
        "cholrot_identity_matvec_s": identity_time,
        "identity_speedup_vs_numpy": numpy_time / identity_time if identity_time else float("inf"),
        "identity_speedup_vs_generic": generic_time / identity_time if identity_time else float("inf"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800, 1600, 3200, 6400])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--upper", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"cholrot backend={backend()}")
    print(f"python={sys.version.split()[0]} numpy={np.__version__} platform={platform.platform()}")
    rows = [one_case(n, seed=n, repeat=args.repeat, lower=not args.upper) for n in args.sizes]
    for row in rows:
        print(
            f"n={row['n']:6d}  "
            f"numpy={row['numpy_cholesky_then_matvec_s']:.6g}s  "
            f"generic={row['cholrot_generic_matvec_s']:.6g}s  "
            f"identity={row['cholrot_identity_matvec_s']:.6g}s  "
            f"speedup={row['identity_speedup_vs_numpy']:.1f}x"
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
