import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import control


def eigenvalue_matrix_np2sp(eval_list):
    """Returns the symbolic eigenvalue matrix from a list
    of eigenvalues from numpy
    """
    return sp.diag(  # Diagonal matrix constructor
        *list(sp.Matrix(eval_list))  # Comma-separated expressions
    ).applyfunc(
        sp.nsimplify
    )  # Uses symbolic numbers


def modal_matrix_np2sp(evec_array):
    """Returns the symbolic modal matrix from a numpy array
    of eigenvectors
    """
    return sp.Matrix(evec_array).applyfunc(sp.nsimplify)  # Uses symbolic numbers


def stability_from_eigenvalues(eval_list):
    """Returns the stability as str of from a list of eigenvalues"""
    real_parts = np.real(eval_list)
    n_zero = 0  # Number of zero real parts
    for l in real_parts:
        if l > 0:
            return "unstable"
        elif l == 0:
            n_zero += 1
    if n_zero > 0:
        return "marginally stable"
    else:
        return "stable"


class StateSpace(control.StateSpace):
    """Subclass of control.StateSpace with extra methods"""

    def eig(self):
        """Returns the eigenvalue matrix Lambda"""
