# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from cholrot import NonPositiveDefiniteError, cholsolve, downdate, logdet_rank1, matvec, update


def spd_case(n=8, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, n))
    A = G.T @ G + n * np.eye(n)
    z = 0.15 * rng.normal(size=n)
    v = rng.normal(size=n)
    B = rng.normal(size=(n, 3))
    return A, z, v, B


@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("alpha", [-1, 1])
@pytest.mark.parametrize("method", ["hy", "hc", "algorithm_a"])
def test_update_matches_numpy(lower, alpha, method):
    A, z, _, _ = spd_case(seed=10 + alpha + int(lower))
    R = np.linalg.cholesky(A) if lower else np.linalg.cholesky(A).T
    D = update(R, z, alpha=alpha, lower=lower, method=method)
    A_new = A + alpha * np.outer(z, z)
    D_ref = np.linalg.cholesky(A_new) if lower else np.linalg.cholesky(A_new).T
    np.testing.assert_allclose(D, D_ref, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("alpha", [-1, 1])
@pytest.mark.parametrize("method", ["hy", "hc", "algorithm_a"])
def test_matvec_matches_materialized_factor_vector(lower, alpha, method):
    A, z, v, _ = spd_case(seed=20 + alpha + int(lower))
    R = np.linalg.cholesky(A) if lower else np.linalg.cholesky(A).T
    D = update(R, z, alpha=alpha, lower=lower, method=method)
    w = matvec(R, z, v, alpha=alpha, lower=lower, method=method)
    np.testing.assert_allclose(w, D @ v, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("alpha", [-1, 1])
@pytest.mark.parametrize("method", ["hy", "hc", "algorithm_a"])
def test_matvec_matches_materialized_factor_matrix(lower, alpha, method):
    A, z, _, B = spd_case(seed=30 + alpha + int(lower))
    R = np.linalg.cholesky(A) if lower else np.linalg.cholesky(A).T
    D = update(R, z, alpha=alpha, lower=lower, method=method)
    W = matvec(R, z, B, alpha=alpha, lower=lower, method=method)
    np.testing.assert_allclose(W, D @ B, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("alpha", [-1, 1])
def test_cholsolve_matches_numpy(lower, alpha):
    A, z, v, B = spd_case(seed=40 + alpha + int(lower))
    R = np.linalg.cholesky(A) if lower else np.linalg.cholesky(A).T
    A_new = A + alpha * np.outer(z, z)
    x = cholsolve(R, z, v, alpha=alpha, lower=lower)
    X = cholsolve(R, z, B, alpha=alpha, lower=lower)
    np.testing.assert_allclose(x, np.linalg.solve(A_new, v), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(X, np.linalg.solve(A_new, B), rtol=2e-12, atol=2e-12)


def test_downdate_alias_and_logdet():
    A, z, _, _ = spd_case(seed=123)
    R = np.linalg.cholesky(A).T
    D = downdate(R, z)
    np.testing.assert_allclose(D.T @ D, A - np.outer(z, z), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        logdet_rank1(R, z, alpha=-1),
        np.linalg.slogdet(A - np.outer(z, z))[1],
        rtol=2e-12,
        atol=2e-12,
    )


def test_non_positive_downdate_raises():
    R = np.eye(3)
    z = np.array([2.0, 0.0, 0.0])
    with pytest.raises(NonPositiveDefiniteError):
        downdate(R, z)
    with pytest.raises(NonPositiveDefiniteError):
        cholsolve(R, z, np.ones(3), alpha=-1)


def test_backend_name_is_stable():
    import cholrot

    assert cholrot.backend() in {"cpp", "python"}
