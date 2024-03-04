import sympy as sp
import numpy as np
import control


class StateSpaceSymbolic:
    """Represents a continuous LTI state-space model in symbolic form"""

    def __init__(self, A, B, C, D, E=None, F=None):
        self.A = sp.Matrix(A)
        self.B = sp.Matrix(B)
        self.C = sp.Matrix(C)
        self.D = sp.Matrix(D)
        if E is None:
            self.E = sp.zeros(*self.B.shape)
        else:
            raise (
                NotImplemented(
                    "Methods not implemented for systems with E and F matrices"
                )
            )
            self.E = sp.Matrix(E)
        if F is None:
            self.F = sp.zeros(*self.D.shape)
        else:
            raise (
                NotImplemented(
                    "Methods not implemented for systems with E and F matrices"
                )
            )
            self.F = sp.Matrix(F)

    def eig(self):
        """Returns L, M: the eigenvalue and eigenvector matrices of A"""
        eig = self.A.eigenvects()
        L = sp.zeros(*self.A.shape)  # Initialize eigenvalue matrix
        M = sp.zeros(*self.A.shape)  # Initialize eigenvector matrix
        k = 0  # Eigen index
        for e in eig:
            val = e[0]
            m = e[1]  # Multiplicity
            for i in range(0, m):
                L[k + i, k + i] = val
            vecs = e[2]
            for i, vec in enumerate(vecs):
                M[:, k + i] = vec
            k = k + m
        return L, M

    def diag_transformation(self, reals_only=False, sort=True, normalize=False):
        """Return (P, D), where D is diagonal and D = P^-1 * self.A * P

        This returns self.A.diagonalize() from SymPy.

        Args:
            reals_only : bool. Whether to throw an error if complex numbers are need
                            to diagonalize. (Default: False)

            sort : bool. Sort the eigenvalues along the diagonal. (Default: True)

            normalize : bool. If True, normalize the columns of P. (Default: False)
        """
        return self.A.diagonalize(reals_only=False, sort=True, normalize=False)

    def is_diagonalizable(self, reals_only=False, **kwargs):
        """Returns ``True`` if self.A is diagonalizable.

        This returns self.A.is_diagonalizable() from SymPy.

        Args:
            - reals_only : bool, optional
                If ``True``, it tests whether the matrix can be diagonalized
                to contain only real numbers on the diagonal.
                If ``False``, it tests whether the matrix can be diagonalized
                at all, even with numbers that may not be real.
        """
        return self.A.is_diagonalizable(reals_only=reals_only)

    def state_transition_matrix(self, t):
        """Returns the state transition matrix"""
        return sp.exp(self.A * t)  # This works for repeated roots, too

    def state_free_response(self, t, x0):
        """Returns the free response of the state vector"""
        x0 = sp.Matrix(x0)
        return self.state_transition_matrix(t) * x0

    def output_free_response(self, t, x0):
        """Returns the free response of the output vector"""
        return self.C * self.state_free_response(t, x0)

    def state_forced_response(self, t, u):
        """Returns the forced response of the state vector"""
        u = sp.Matrix(u)
        Phi = self.state_transition_matrix(t)
        tau = sp.symbol("tau", real=True)
        x_fo = Phi * sp.integrate(
            Phi.subs(t, -tau) * self.B * u.subs(t, tau), (tau, 0, t)
        )
        return x_fo.simplify

    def output_forced_response(self, t, u):
        """Returns the forced response of the output vector"""
        return self.C * self.state_forced_response(t, u)

    def state_response(self, t, x0=None, u=None):
        """Returns the state response for initial condition x0 and input u"""
        if x0 is None and u is None:
            return sp.zeros(*self.A.shape[0], 1)
        elif x0 is None:
            return self.state_forced_response(t, u)
        elif u is None:
            return self.state_free_response(t, x0)
        else:
            return self.state_free_response(t, x0) + self.state_forced_response(t, u)

    def output_response(self, t, x0=None, u=None):
        """Returns the output response for initial condition x0 and input u"""
        if x0 is None and u is None:
            return sp.zeros(*self.C.shape[0], 1)
        elif x0 is None:
            return self.output_forced_response(t, u)
        elif u is None:
            return self.output_free_response(t, x0)
        else:
            return self.output_free_response(t, x0) + self.output_forced_response(t, u)

    def to_numpy(self, params: dict = {}):
        """Returns A, B, C, D as NumPy arrays"""
        A = np.array(self.A.subs(params)).astype(np.float64)
        B = np.array(self.B.subs(params)).astype(np.float64)
        C = np.array(self.C.subs(params)).astype(np.float64)
        D = np.array(self.D.subs(params)).astype(np.float64)
        return A, B, C, D

    def to_control(self, params: dict = {}):
        """Returns an equivalent Control Systems package control.StateSpace object"""
        A, B, C, D = self.to_numpy(params=params)
        return control.ss(A, B, C, D)
