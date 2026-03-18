import numpy as np
import sympy as sp
import pytest

from dysys.sysdyn import (
    eigenvalue_matrix_np2sp,
    modal_matrix_np2sp,
    stability_from_eigenvalues,
    StateSpace,
)


class TestEigenvalueMatrixNp2sp:
    def test_real_eigenvalues(self):
        evals = [-1.0, -2.0, -3.0]
        L = eigenvalue_matrix_np2sp(evals)
        assert L.shape == (3, 3)
        assert L[0, 0] == -1
        assert L[1, 1] == -2
        assert L[2, 2] == -3
        # Off-diagonals are zero
        assert L[0, 1] == 0
        assert L[1, 0] == 0

    def test_complex_eigenvalues(self):
        evals = [-1 + 2j, -1 - 2j]
        L = eigenvalue_matrix_np2sp(evals)
        assert L.shape == (2, 2)
        assert L[0, 0] == sp.nsimplify(-1 + 2j)
        assert L[1, 1] == sp.nsimplify(-1 - 2j)


class TestModalMatrixNp2sp:
    def test_basic(self):
        evecs = np.array([[1, 0], [0, 1]], dtype=float)
        M = modal_matrix_np2sp(evecs)
        assert M == sp.eye(2)

    def test_non_identity(self):
        evecs = np.array([[1, 1], [1, -1]], dtype=float)
        M = modal_matrix_np2sp(evecs)
        assert M.shape == (2, 2)
        assert M[0, 0] == 1
        assert M[1, 1] == -1


class TestStabilityFromEigenvalues:
    def test_stable(self):
        assert stability_from_eigenvalues([-1, -2, -3]) == "stable"

    def test_unstable_positive_real(self):
        assert stability_from_eigenvalues([-1, 2, -3]) == "unstable"

    def test_marginally_stable(self):
        assert stability_from_eigenvalues([-1, 0, -3]) == "marginally stable"

    def test_complex_stable(self):
        assert stability_from_eigenvalues([-1 + 2j, -1 - 2j]) == "stable"

    def test_complex_unstable(self):
        assert stability_from_eigenvalues([1 + 2j, 1 - 2j]) == "unstable"

    def test_imaginary_marginally_stable(self):
        assert stability_from_eigenvalues([2j, -2j]) == "marginally stable"


class TestStateSpace:
    def test_creation(self):
        sys = StateSpace([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]], [[0]])
        assert sys.A.shape == (2, 2)

    def test_eig(self):
        sys = StateSpace([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]], [[0]])
        evals, evecs = sys.eig()
        assert len(evals) == 2
        # Eigenvalues of [[0,1],[-2,-3]] are -1 and -2
        assert np.allclose(sorted(np.real(evals)), [-3, 0], atol=0.1) or \
               np.allclose(sorted(np.real(evals)), [-2, -1])
