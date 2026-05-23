# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import Literal

import numpy as np

try:
    if os.environ.get("CHOLROT_PURE_PYTHON") == "1":
        raise ImportError("pure Python backend requested")
    from . import _core_ext
except Exception:  # pragma: no cover - depends on build environment
    _core_ext = None

Method = Literal["hy", "hc", "algorithm_a", "a"]


class CholrotError(ValueError):
    """Base exception for invalid Cholesky rank-one operations."""


class NonPositiveDefiniteError(CholrotError):
    """Raised when an update/downdate would not remain positive definite."""


def _normalise_method(method: str) -> str:
    m = method.lower().replace("-", "_").replace(" ", "_")
    aliases = {"alg_a": "algorithm_a", "alga": "algorithm_a", "a": "algorithm_a"}
    m = aliases.get(m, m)
    if m not in {"hy", "hc", "algorithm_a"}:
        raise ValueError("method must be one of {'hy', 'hc', 'algorithm_a'}")
    return m


def _normalise_alpha(alpha: float) -> float:
    if alpha not in (-1, 1, -1.0, 1.0):
        raise ValueError("alpha must be +1 for an update or -1 for a downdate")
    return float(alpha)


def _as_factor_and_vector(R: np.ndarray, z: np.ndarray, *, overwrite: bool) -> tuple[np.ndarray, np.ndarray]:
    R_arr = np.asarray(R)
    z_arr = np.asarray(z)
    if R_arr.ndim != 2 or R_arr.shape[0] != R_arr.shape[1]:
        raise ValueError("R must be a square 2D array")
    n = R_arr.shape[0]
    if z_arr.ndim != 1 or z_arr.shape[0] != n:
        raise ValueError("z must be a one-dimensional array with length R.shape[0]")
    dtype = np.result_type(R_arr.dtype, z_arr.dtype, np.float64)
    R_work = R_arr.astype(dtype, copy=not overwrite)
    z_work = z_arr.astype(dtype, copy=True)
    if not np.all(np.isfinite(R_work)) or not np.all(np.isfinite(z_work)):
        raise ValueError("R and z must contain only finite values")
    if np.any(np.diag(R_work) <= 0):
        raise ValueError("R must have a positive diagonal")
    return R_work, z_work


def _as_rhs(v: np.ndarray, n: int, dtype: np.dtype) -> tuple[np.ndarray, bool]:
    arr = np.asarray(v, dtype=dtype)
    vector = arr.ndim == 1
    if vector:
        if arr.shape[0] != n:
            raise ValueError("v must have length R.shape[0]")
        return arr, True
    if arr.ndim == 2 and arr.shape[0] == n:
        return arr, False
    raise ValueError("v must be a vector of length n or a matrix with shape (n, k)")


def _check_triangle(R: np.ndarray, *, lower: bool, tol: float = 1e-12) -> None:
    stray = np.triu(R, 1) if lower else np.tril(R, -1)
    if np.max(np.abs(stray), initial=0.0) > tol:
        kind = "lower" if lower else "upper"
        raise ValueError(f"R must be {kind} triangular")


def update(
    R: np.ndarray,
    z: np.ndarray,
    *,
    alpha: float = 1,
    lower: bool = False,
    method: Method = "hy",
    overwrite: bool = False,
    check: bool = True,
) -> np.ndarray:
    """Return the rank-one modified Cholesky factor.

    Parameters
    ----------
    R:
        Cholesky factor. If ``lower=False`` (default), ``R`` is upper triangular
        and represents ``A = R.T @ R``. If ``lower=True``, ``R`` is lower
        triangular and represents ``A = R @ R.T``.
    z:
        Rank-one modification vector.
    alpha:
        ``+1`` computes an update, ``A + z z.T``. ``-1`` computes a downdate,
        ``A - z z.T``.
    lower:
        Whether ``R`` is lower triangular.
    method:
        ``"hy"`` for the hyperbolic rotation form, ``"hc"`` for the Chambers
        variant, or ``"algorithm_a"`` for Algorithm A.
    overwrite:
        If true, inputs may be reused internally. The public return value is the
        modified factor; ``z`` is never modified.
    check:
        Check that input is triangular.

    Returns
    -------
    numpy.ndarray
        The modified Cholesky factor ``D`` with the same triangular orientation
        as ``R``.
    """
    alpha = _normalise_alpha(alpha)
    method = _normalise_method(method)
    R_work, z_work = _as_factor_and_vector(R, z, overwrite=overwrite)
    if check:
        _check_triangle(R_work, lower=lower)
    if _core_ext is not None:
        try:
            return _core_ext.update_c(
                np.ascontiguousarray(R_work, dtype=np.float64),
                np.ascontiguousarray(z_work, dtype=np.float64),
                alpha,
                lower,
                method,
            )
        except (RuntimeError, ValueError) as exc:
            if "positive definite" in str(exc):
                raise NonPositiveDefiniteError(str(exc)) from exc
            raise
    return _update_python(R_work, z_work, alpha=alpha, lower=lower, method=method)


def rank1_update(R: np.ndarray, z: np.ndarray, **kwargs) -> np.ndarray:
    """Return the Cholesky factor after ``A + z z.T``."""
    return update(R, z, alpha=1, **kwargs)


def downdate(R: np.ndarray, z: np.ndarray, **kwargs) -> np.ndarray:
    """Return the Cholesky factor after ``A - z z.T``."""
    return update(R, z, alpha=-1, **kwargs)


def matvec(
    R: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    *,
    alpha: float = -1,
    lower: bool = False,
    method: Method = "hy",
    check: bool = True,
) -> np.ndarray:
    """Compute ``D @ v`` without materializing the modified factor ``D``.

    ``D`` is the Cholesky factor obtained by applying ``A + alpha*z*z.T`` to the
    matrix represented by ``R``. This function fuses the rank-one modification
    with triangular matrix-vector/matrix multiplication.

    Parameters are the same as :func:`update`, with ``v`` supplied as a vector or
    a dense matrix of right-hand sides. The result has the same shape as ``v``.
    """
    alpha = _normalise_alpha(alpha)
    method = _normalise_method(method)
    R_work, z_work = _as_factor_and_vector(R, z, overwrite=False)
    if check:
        _check_triangle(R_work, lower=lower)
    rhs, vector = _as_rhs(v, R_work.shape[0], R_work.dtype)
    rhs2 = rhs[:, None] if vector else rhs
    if _core_ext is not None:
        try:
            out = _core_ext.matvec_c(
                np.ascontiguousarray(R_work, dtype=np.float64),
                np.ascontiguousarray(z_work, dtype=np.float64),
                np.ascontiguousarray(rhs2, dtype=np.float64),
                alpha,
                lower,
                method,
            )
        except (RuntimeError, ValueError) as exc:
            if "positive definite" in str(exc):
                raise NonPositiveDefiniteError(str(exc)) from exc
            raise
    else:
        out = _matvec_python(R_work, z_work, rhs2, alpha=alpha, lower=lower, method=method)
    return out[:, 0] if vector else out



def identity_matvec(
    z: np.ndarray,
    v: np.ndarray,
    *,
    alpha: float = -1,
    lower: bool = True,
) -> np.ndarray:
    """Compute ``D @ v`` for ``D D.T = I + alpha*z*z.T`` in linear time.

    This is the structured identity-factor kernel used in the hyperbolic
    rotation benchmarks. It does not materialize ``I + alpha*z*z.T`` or the
    Cholesky factor ``D``.  For ``lower=False`` it returns the product with the
    corresponding upper factor ``D.T``.
    """
    alpha = _normalise_alpha(alpha)
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 1:
        raise ValueError("z must be one-dimensional")
    rhs, vector = _as_rhs(v, z_arr.shape[0], z_arr.dtype)
    rhs2 = rhs[:, None] if vector else rhs
    if not np.all(np.isfinite(z_arr)) or not np.all(np.isfinite(rhs2)):
        raise ValueError("z and v must contain only finite values")

    if _core_ext is not None:
        try:
            out = _core_ext.identity_matvec_c(
                np.ascontiguousarray(z_arr, dtype=np.float64),
                np.ascontiguousarray(rhs2, dtype=np.float64),
                alpha,
                lower,
            )
        except (RuntimeError, ValueError) as exc:
            if "positive definite" in str(exc):
                raise NonPositiveDefiniteError(str(exc)) from exc
            raise
    else:
        out = _identity_matvec_python(z_arr, rhs2, alpha=alpha, lower=lower)
    return out[:, 0] if vector else out


def _identity_matvec_python(z: np.ndarray, V: np.ndarray, *, alpha: float, lower: bool) -> np.ndarray:
    n, k = V.shape
    W = np.empty_like(V, dtype=np.float64)
    q = 1.0
    if lower:
        prefix = np.zeros(k, dtype=np.float64)
        for i in range(n):
            zi = float(z[i])
            q_next = q + alpha * zi * zi
            if q_next <= 0.0 or not np.isfinite(q_next):
                raise NonPositiveDefiniteError("rank-one operation is not positive definite")
            denom = np.sqrt(q * q_next)
            W[i, :] = np.sqrt(q_next / q) * V[i, :] + alpha * zi * prefix
            prefix += zi * V[i, :] / denom
            q = q_next
    else:
        suffix = z @ V
        for i in range(n):
            zi = float(z[i])
            suffix = suffix - zi * V[i, :]
            q_next = q + alpha * zi * zi
            if q_next <= 0.0 or not np.isfinite(q_next):
                raise NonPositiveDefiniteError("rank-one operation is not positive definite")
            W[i, :] = np.sqrt(q_next / q) * V[i, :] + alpha * zi * suffix / np.sqrt(q * q_next)
            q = q_next
    return W

def solve_rank1(
    R: np.ndarray,
    z: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float = -1,
    lower: bool = False,
    check: bool = True,
) -> np.ndarray:
    """Solve ``(A + alpha*z*z.T) x = b`` without forming the modified matrix.

    The base SPD matrix ``A`` is represented by the Cholesky factor ``R``:

    * ``A = R.T @ R`` for an upper factor (``lower=False``), or
    * ``A = R @ R.T`` for a lower factor (``lower=True``).

    The computation uses the rank-one inverse identity and triangular solves
    with the original factor. It does **not** materialize ``A + alpha*z*z.T`` or
    the modified Cholesky factor.
    """
    alpha = _normalise_alpha(alpha)
    R_work, z_work = _as_factor_and_vector(R, z, overwrite=False)
    if check:
        _check_triangle(R_work, lower=lower)
    rhs, vector = _as_rhs(b, R_work.shape[0], R_work.dtype)
    rhs2 = rhs[:, None] if vector else rhs

    if _core_ext is not None:
        try:
            out = _core_ext.solve_rank1_c(
                np.ascontiguousarray(R_work, dtype=np.float64),
                np.ascontiguousarray(z_work, dtype=np.float64),
                np.ascontiguousarray(rhs2, dtype=np.float64),
                alpha,
                lower,
            )
        except (RuntimeError, ValueError) as exc:
            if "positive definite" in str(exc):
                raise NonPositiveDefiniteError(str(exc)) from exc
            raise
        return out[:, 0] if vector else out

    Ainv_b = _spd_cholesky_solve(R_work, rhs2, lower=lower)
    Ainv_z = _spd_cholesky_solve(R_work, z_work[:, None], lower=lower)
    zAinv_b = z_work @ Ainv_b
    denom = 1.0 + alpha * float(z_work @ Ainv_z[:, 0])
    if denom <= 0:
        raise NonPositiveDefiniteError("rank-one downdate/update is not positive definite")
    out = Ainv_b - alpha * Ainv_z @ (zAinv_b[None, :] / denom)
    return out[:, 0] if vector else out


# Friendly alias used in the README.
cholsolve = solve_rank1


def logdet_rank1(
    R: np.ndarray,
    z: np.ndarray,
    *,
    alpha: float = -1,
    lower: bool = False,
    check: bool = True,
) -> float:
    """Return ``log(det(A + alpha*z*z.T))`` from a Cholesky factor of ``A``.

    This uses the matrix determinant lemma and triangular solves with ``R``.
    """
    alpha = _normalise_alpha(alpha)
    R_work, z_work = _as_factor_and_vector(R, z, overwrite=False)
    if check:
        _check_triangle(R_work, lower=lower)
    Ainv_z = _spd_cholesky_solve(R_work, z_work[:, None], lower=lower)[:, 0]
    correction = 1.0 + alpha * float(z_work @ Ainv_z)
    if correction <= 0:
        raise NonPositiveDefiniteError("rank-one downdate/update is not positive definite")
    return 2.0 * float(np.sum(np.log(np.diag(R_work)))) + float(np.log(correction))


def backend() -> str:
    """Return the active kernel backend: ``"cpp"`` or ``"python"``."""
    return "cpp" if _core_ext is not None else "python"


def _update_python(R: np.ndarray, z: np.ndarray, *, alpha: float, lower: bool, method: str) -> np.ndarray:
    if method == "hy":
        return _factor_hy(R, z, alpha=alpha, lower=lower)
    if method == "hc":
        return _factor_hc(R, z, alpha=alpha, lower=lower)
    return _factor_algorithm_a(R, z, alpha=alpha, lower=lower)


def _matvec_python(R: np.ndarray, z: np.ndarray, V: np.ndarray, *, alpha: float, lower: bool, method: str) -> np.ndarray:
    if method == "hy":
        return _matvec_hy(R, z, V, alpha=alpha, lower=lower)
    if method == "hc":
        return _matvec_hc(R, z, V, alpha=alpha, lower=lower)
    return _matvec_algorithm_a(R, z, V, alpha=alpha, lower=lower)


def _factor_hy(R: np.ndarray, z: np.ndarray, *, alpha: float, lower: bool) -> np.ndarray:
    n = R.shape[0]
    D = np.zeros_like(R)
    for i in range(n):
        disc = R[i, i] * R[i, i] + alpha * z[i] * z[i]
        if disc <= 0:
            raise NonPositiveDefiniteError("rank-one operation is not positive definite")
        dii = np.sqrt(disc)
        ch = R[i, i] / dii
        sh = z[i] / dii
        D[i, i] = dii
        if lower:
            for j in range(i + 1, n):
                dji = ch * R[j, i] + alpha * sh * z[j]
                z[j] = -sh * R[j, i] + ch * z[j]
                D[j, i] = dji
        else:
            for j in range(i + 1, n):
                dij = ch * R[i, j] + alpha * sh * z[j]
                z[j] = -sh * R[i, j] + ch * z[j]
                D[i, j] = dij
    return D


def _factor_hc(R: np.ndarray, z: np.ndarray, *, alpha: float, lower: bool) -> np.ndarray:
    n = R.shape[0]
    D = np.zeros_like(R)
    for i in range(n):
        s = z[i] / R[i, i]
        c2 = 1.0 + alpha * s * s
        if c2 <= 0:
            raise NonPositiveDefiniteError("rank-one operation is not positive definite")
        c = np.sqrt(c2)
        D[i, i] = c * R[i, i]
        if lower:
            for j in range(i + 1, n):
                dji = (R[j, i] + alpha * s * z[j]) / c
                z[j] = -s * dji + c * z[j]
                D[j, i] = dji
        else:
            for j in range(i + 1, n):
                dij = (R[i, j] + alpha * s * z[j]) / c
                z[j] = -s * dij + c * z[j]
                D[i, j] = dij
    return D


def _factor_algorithm_a(R: np.ndarray, z: np.ndarray, *, alpha: float, lower: bool) -> np.ndarray:
    n = R.shape[0]
    D = np.zeros_like(R)
    lam_prev = 1.0
    for i in range(n):
        a_i = z[i] / R[i, i]
        lam_next = lam_prev + alpha * a_i * a_i
        if lam_next <= 0:
            raise NonPositiveDefiniteError("rank-one operation is not positive definite")
        beta = np.sqrt(lam_next)
        beta_prev = np.sqrt(lam_prev)
        scale = beta / beta_prev
        D[i, i] = scale * R[i, i]
        if lower:
            for j in range(i + 1, n):
                z[j] = z[j] - a_i * R[j, i]
                D[j, i] = scale * R[j, i] + alpha * (a_i / (beta_prev * beta)) * z[j]
        else:
            for j in range(i + 1, n):
                z[j] = z[j] - a_i * R[i, j]
                D[i, j] = scale * R[i, j] + alpha * (a_i / (beta_prev * beta)) * z[j]
        lam_prev = lam_next
    return D


def _matvec_hy(R: np.ndarray, z: np.ndarray, V: np.ndarray, *, alpha: float, lower: bool) -> np.ndarray:
    n = R.shape[0]
    W = np.zeros((n, V.shape[1]), dtype=np.result_type(R, V, np.float64))
    for i in range(n):
        disc = R[i, i] * R[i, i] + alpha * z[i] * z[i]
        if disc <= 0:
            raise NonPositiveDefiniteError("rank-one operation is not positive definite")
        dii = np.sqrt(disc)
        ch = R[i, i] / dii
        sh = z[i] / dii
        if lower:
            W[i] += dii * V[i]
            for j in range(i + 1, n):
                dji = ch * R[j, i] + alpha * sh * z[j]
                z[j] = -sh * R[j, i] + ch * z[j]
                W[j] += dji * V[i]
        else:
            W[i] += dii * V[i]
            for j in range(i + 1, n):
                dij = ch * R[i, j] + alpha * sh * z[j]
                z[j] = -sh * R[i, j] + ch * z[j]
                W[i] += dij * V[j]
    return W


def _matvec_hc(R: np.ndarray, z: np.ndarray, V: np.ndarray, *, alpha: float, lower: bool) -> np.ndarray:
    n = R.shape[0]
    W = np.zeros((n, V.shape[1]), dtype=np.result_type(R, V, np.float64))
    for i in range(n):
        s = z[i] / R[i, i]
        c2 = 1.0 + alpha * s * s
        if c2 <= 0:
            raise NonPositiveDefiniteError("rank-one operation is not positive definite")
        c = np.sqrt(c2)
        dii = c * R[i, i]
        if lower:
            W[i] += dii * V[i]
            for j in range(i + 1, n):
                dji = (R[j, i] + alpha * s * z[j]) / c
                z[j] = -s * dji + c * z[j]
                W[j] += dji * V[i]
        else:
            W[i] += dii * V[i]
            for j in range(i + 1, n):
                dij = (R[i, j] + alpha * s * z[j]) / c
                z[j] = -s * dij + c * z[j]
                W[i] += dij * V[j]
    return W


def _matvec_algorithm_a(R: np.ndarray, z: np.ndarray, V: np.ndarray, *, alpha: float, lower: bool) -> np.ndarray:
    n = R.shape[0]
    W = np.zeros((n, V.shape[1]), dtype=np.result_type(R, V, np.float64))
    lam_prev = 1.0
    for i in range(n):
        a_i = z[i] / R[i, i]
        lam_next = lam_prev + alpha * a_i * a_i
        if lam_next <= 0:
            raise NonPositiveDefiniteError("rank-one operation is not positive definite")
        beta = np.sqrt(lam_next)
        beta_prev = np.sqrt(lam_prev)
        scale = beta / beta_prev
        dii = scale * R[i, i]
        if lower:
            W[i] += dii * V[i]
            for j in range(i + 1, n):
                z[j] = z[j] - a_i * R[j, i]
                dji = scale * R[j, i] + alpha * (a_i / (beta_prev * beta)) * z[j]
                W[j] += dji * V[i]
        else:
            W[i] += dii * V[i]
            for j in range(i + 1, n):
                z[j] = z[j] - a_i * R[i, j]
                dij = scale * R[i, j] + alpha * (a_i / (beta_prev * beta)) * z[j]
                W[i] += dij * V[j]
        lam_prev = lam_next
    return W


def _triangular_solve(T: np.ndarray, B: np.ndarray, *, lower: bool, trans: bool = False) -> np.ndarray:
    M = T.T if trans else T
    is_lower = (not lower) if trans else lower
    n = M.shape[0]
    X = np.empty_like(B, dtype=np.result_type(T, B, np.float64))
    if is_lower:
        for i in range(n):
            rhs = B[i] - M[i, :i] @ X[:i]
            X[i] = rhs / M[i, i]
    else:
        for i in range(n - 1, -1, -1):
            rhs = B[i] - M[i, i + 1 :] @ X[i + 1 :]
            X[i] = rhs / M[i, i]
    return X


def _spd_cholesky_solve(R: np.ndarray, B: np.ndarray, *, lower: bool) -> np.ndarray:
    if lower:
        y = _triangular_solve(R, B, lower=True, trans=False)
        return _triangular_solve(R, y, lower=True, trans=True)
    y = _triangular_solve(R, B, lower=False, trans=True)
    return _triangular_solve(R, y, lower=False, trans=False)
