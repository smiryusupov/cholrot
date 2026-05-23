# SPDX-License-Identifier: Apache-2.0
"""Rank-one Cholesky modifications and implicit factor operations.

`cholrot` focuses on a small NumPy API around Cholesky rank-one updates and
 downdates.  In addition to materializing the modified Cholesky factor, it can
compute products with the modified factor directly and solve rank-one modified
SPD systems without forming the modified matrix.
"""

from .core import (
    CholrotError,
    backend,
    NonPositiveDefiniteError,
    cholsolve,
    downdate,
    logdet_rank1,
    matvec,
    identity_matvec,
    rank1_update,
    solve_rank1,
    update,
)

__all__ = [
    "CholrotError",
    "backend",
    "NonPositiveDefiniteError",
    "update",
    "downdate",
    "rank1_update",
    "matvec",
    "identity_matvec",
    "solve_rank1",
    "cholsolve",
    "logdet_rank1",
]

__version__ = "0.1.0"
