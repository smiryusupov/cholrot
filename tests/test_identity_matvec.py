# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from cholrot import NonPositiveDefiniteError, identity_matvec


@pytest.mark.parametrize("alpha", [-1, 1])
@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("matrix_rhs", [False, True])
def test_identity_matvec_matches_numpy(alpha, lower, matrix_rhs):
    rng = np.random.default_rng(1234)
    n = 12
    z = 0.1 * rng.normal(size=n)
    if alpha == -1:
        z *= 0.5 / np.linalg.norm(z)
    V = rng.normal(size=(n, 3)) if matrix_rhs else rng.normal(size=n)

    L = np.linalg.cholesky(np.eye(n) + alpha * np.outer(z, z))
    D = L if lower else L.T
    np.testing.assert_allclose(identity_matvec(z, V, alpha=alpha, lower=lower), D @ V, rtol=1e-12, atol=1e-12)


def test_identity_downdate_rejects_non_spd():
    z = np.array([0.9, 0.9])
    v = np.ones(2)
    with pytest.raises(NonPositiveDefiniteError):
        identity_matvec(z, v, alpha=-1)
