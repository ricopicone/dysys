import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import control as ct

def eigenvalue_matrix_sym(eval_list):
    """Returns the symbolic eigenvalue matrix from a list 
        of eigenvalues
    """
    return sp.diag(             # Diagonal matrix constructor
        *list(sp.Matrix(eval_list)) # Comma-separated expressions
    ).applyfunc(sp.nsimplify)   # Uses symbolic numbers

def modal_matrix_sym(evec_array):
    """Returns the symbolic modal matrix from a numpy array
        of eigenvectors
    """
    return sp.Matrix(
        M_
    ).applyfunc(sp.nsimplify)   # Uses symbolic numbers