import sympy as sp
import numpy as np
import pytest

from dysys.statespacesymbolic import StateSpaceSymbolic, sss


class TestStateSpaceSymbolicInit:
    def test_basic_creation(self):
        A = [[-1, 0], [0, -2]]
        B = [[1], [0]]
        C = [[1, 0]]
        D = [[0]]
        sys = StateSpaceSymbolic(A, B, C, D)
        assert sys.A == sp.Matrix(A)
        assert sys.B == sp.Matrix(B)
        assert sys.C == sp.Matrix(C)
        assert sys.D == sp.Matrix(D)
        assert sys.E == sp.zeros(2, 1)
        assert sys.F == sp.zeros(1, 1)

    def test_e_matrix_raises(self):
        with pytest.raises(NotImplementedError):
            StateSpaceSymbolic([[1]], [[1]], [[1]], [[0]], E=[[1]])

    def test_f_matrix_raises(self):
        with pytest.raises(NotImplementedError):
            StateSpaceSymbolic([[1]], [[1]], [[1]], [[0]], F=[[1]])

    def test_sss_factory(self):
        sys = sss([[-1]], [[1]], [[1]], [[0]])
        assert isinstance(sys, StateSpaceSymbolic)

    def test_sss_passes_e_f(self):
        with pytest.raises(NotImplementedError):
            sss([[1]], [[1]], [[1]], [[0]], E=[[1]])


class TestStateSpaceSymbolicEig:
    def test_diagonal_system(self):
        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        L, M = sys.eig()
        # Eigenvalues should be -1 and -2
        eigenvalues = {L[i, i] for i in range(2)}
        assert eigenvalues == {sp.Integer(-1), sp.Integer(-2)}

    def test_coupled_system(self):
        sys = StateSpaceSymbolic([[-4, -3], [0, -8]], [[0], [1]], [[1, 0]], [[0]])
        L, M = sys.eig()
        eigenvalues = {L[i, i] for i in range(2)}
        assert eigenvalues == {sp.Integer(-4), sp.Integer(-8)}


class TestDiagTransformation:
    def test_uses_parameters(self):
        A = [[-1, 0], [0, -2]]
        sys = StateSpaceSymbolic(A, [[1], [0]], [[1, 0]], [[0]])
        P, D = sys.diag_transformation()
        # D should be diagonal with eigenvalues
        assert D[0, 1] == 0
        assert D[1, 0] == 0
        # P^-1 * A * P == D
        assert sp.simplify(P.inv() * sys.A * P - D) == sp.zeros(2)

    def test_is_diagonalizable(self):
        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        assert sys.is_diagonalizable() is True


class TestStateTransitionMatrix:
    def test_diagonal(self):
        t = sp.Symbol("t", positive=True)
        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        Phi = sys.state_transition_matrix(t)
        assert Phi[0, 0] == sp.exp(-t)
        assert Phi[1, 1] == sp.exp(-2 * t)
        assert Phi[0, 1] == 0
        assert Phi[1, 0] == 0


class TestFreeResponse:
    def test_state_free_response(self):
        t = sp.Symbol("t", positive=True)
        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        x0 = [1, 2]
        x = sys.state_free_response(t, x0)
        assert x[0] == sp.exp(-t)
        assert x[1] == 2 * sp.exp(-2 * t)

    def test_output_free_response(self):
        t = sp.Symbol("t", positive=True)
        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        y = sys.output_free_response(t, [1, 2])
        assert y[0] == sp.exp(-t)


class TestResponseDefaults:
    def test_state_response_no_input(self):
        t = sp.Symbol("t", positive=True)
        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        x = sys.state_response(t)
        assert x == sp.zeros(2, 1)

    def test_output_response_no_input(self):
        t = sp.Symbol("t", positive=True)
        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        y = sys.output_response(t)
        assert y == sp.zeros(1, 1)

    def test_state_response_free_only(self):
        t = sp.Symbol("t", positive=True)
        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        x = sys.state_response(t, x0=[1, 0])
        assert x[0] == sp.exp(-t)


class TestToNumpy:
    def test_basic(self):
        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        A, B, C, D = sys.to_numpy()
        assert isinstance(A, np.ndarray)
        np.testing.assert_array_almost_equal(A, [[-1, 0], [0, -2]])

    def test_with_params(self):
        a = sp.Symbol("a")
        sys = StateSpaceSymbolic([[-a, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        A, B, C, D = sys.to_numpy(params={a: 3})
        np.testing.assert_array_almost_equal(A, [[-3, 0], [0, -2]])


class TestToControl:
    def test_basic(self):
        import control

        sys = StateSpaceSymbolic([[-1, 0], [0, -2]], [[1], [0]], [[1, 0]], [[0]])
        css = sys.to_control()
        assert isinstance(css, control.StateSpace)
