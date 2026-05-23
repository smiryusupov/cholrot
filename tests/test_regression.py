# SPDX-License-Identifier: Apache-2.0
"""Golden-value regression tests for the public numerical API.

These tests complement property tests that compare against NumPy.  They protect
against accidental sign-convention changes, lower/upper orientation changes, and
silent changes in the algorithmic recurrences.
"""

from __future__ import annotations

import numpy as np

from cholrot import cholsolve, downdate, logdet_rank1, matvec, update


R = np.array(
    [
        [3.2807423700102474, -0.3555956338607549, 1.0832258061181794, 0.5903186687374778, 0.9501810135497731],
        [0.0, 2.400597233861356, 0.4181102522262448, 1.026464596441424, -0.36601498322397574],
        [0.0, 0.0, 2.0641781569952804, 0.19464122750518473, 0.371308536994091],
        [0.0, 0.0, 0.0, 2.7318169849408616, -0.6225259810295224],
        [0.0, 0.0, 0.0, 0.0, 1.8478048839533665],
    ]
)

L = R.T
z = np.array([-0.2783304691082159, -0.20384131729147562, -0.06025064380902171, 0.12049081024416804, -0.1220164792759453])
v = np.array([0.2665934882837246, -0.5864332185019752, 2.0428442694913076, -2.1304051288848593, -0.32865056524117653])
B = np.array(
    [
        [0.11197012453083513, 0.057759030115997506],
        [-0.8669690870249361, 0.9506562850976685],
        [0.04671082154195698, -0.4075739602609285],
        [-0.40166445295457576, -0.9787003829057713],
        [-0.9403820144435925, -1.847908300659203],
    ]
)


EXPECTED_DOWNDATE_HY_UPPER = np.array(
    [
        [3.2689145978973597, -0.3742382602622567, 1.0820151773741866, 0.6027137366454498, 0.9432299664235609],
        [0.0, 2.3890813665432957, 0.42324797678268333, 1.0482409307925162, -0.3718642850434576],
        [0.0, 0.0, 2.062886317729452, 0.18510294978625053, 0.3742925811110452],
        [0.0, 0.0, 0.0, 2.718816086159746, -0.6166000077806459],
        [0.0, 0.0, 0.0, 0.0, 1.8475546268468668],
    ]
)

EXPECTED_UPDATE_HC_LOWER = np.array(
    [
        [3.2925276534016312, 0.0, 0.0, 0.0, 0.0],
        [-0.33709129579689207, 2.4118947064448566, 0.0, 0.0, 0.0],
        [1.084441731168752, 0.4131031856895224, 2.0654267897037153, 0.0, 0.0],
        [0.5780201124902892, 1.0052254215078917, 0.203854891010656, 2.7443015077318074, 0.0],
        [0.9570944714745664, -0.3603119814047164, 0.3684262090638778, -0.6282033388677308, 1.8480418300927273],
    ]
)

EXPECTED_MATVEC_DOWNDATE_HY_UPPER = np.array([1.707308100192355, -2.6473714188956645, 3.696799750841566, -5.589533793364559, -0.6071998724271738])

EXPECTED_MATVEC_UPDATE_HC_LOWER_MATRIX = np.array(
    [
        [0.36866473137259903, 0.19017320389057943],
        [-2.1287823060154123, 2.2734127953698255],
        [-0.14024483387711095, -0.38645873382932516],
        [-1.8995445163731486, -1.7799251356923693],
        [-1.0487835208837408, -3.237601919909962],
    ]
)

EXPECTED_CHOLSOLVE_DOWNDATE_UPPER = np.array([0.0022973634427880373, -0.09781736534348683, 0.5877217326144497, -0.3761674043450571, -0.39414070901823034])
EXPECTED_LOGDET_DOWNDATE_UPPER = 8.787065041808463


def test_golden_downdate_hy_upper():
    np.testing.assert_allclose(downdate(R, z, method="hy"), EXPECTED_DOWNDATE_HY_UPPER, rtol=1e-13, atol=1e-13)


def test_golden_algorithm_a_matches_hy_upper():
    # Algorithm A and HY are mathematically equivalent here but exercise different recurrences.
    np.testing.assert_allclose(downdate(R, z, method="algorithm_a"), EXPECTED_DOWNDATE_HY_UPPER, rtol=1e-13, atol=1e-13)


def test_golden_update_hc_lower():
    np.testing.assert_allclose(update(L, z, alpha=1, lower=True, method="hc"), EXPECTED_UPDATE_HC_LOWER, rtol=1e-13, atol=1e-13)


def test_golden_matvec_downdate_hy_upper():
    np.testing.assert_allclose(matvec(R, z, v, method="hy"), EXPECTED_MATVEC_DOWNDATE_HY_UPPER, rtol=1e-13, atol=1e-13)


def test_golden_matvec_update_hc_lower_matrix():
    np.testing.assert_allclose(
        matvec(L, z, B, alpha=1, lower=True, method="hc"),
        EXPECTED_MATVEC_UPDATE_HC_LOWER_MATRIX,
        rtol=1e-13,
        atol=1e-13,
    )


def test_golden_cholsolve_and_logdet():
    np.testing.assert_allclose(cholsolve(R, z, v, alpha=-1), EXPECTED_CHOLSOLVE_DOWNDATE_UPPER, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(logdet_rank1(R, z, alpha=-1), EXPECTED_LOGDET_DOWNDATE_UPPER, rtol=1e-13, atol=1e-13)
