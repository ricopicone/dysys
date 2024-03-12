from control import TransferFunction
import numpy as np

def extract_poles_and_zeros(sys):
    """
    Extract poles and zeros from a TransferFunction object.
    """
    poles = np.roots(sys.den[0][0])
    zeros = np.roots(sys.num[0][0])
    return np.sort_complex(poles), np.sort_complex(zeros)

def factor_poles(poles):
    """
    Factor poles into sub-TF models.
    """
    F = []
    F_gain = 1

    for pole in poles:
        if np.iscomplex(pole):
            F.append(TransferFunction([1], [1, -2 * np.real(pole), np.abs(pole)**2 + np.imag(pole)**2]))
            F_gain /= np.abs(pole)**2
        else:
            F.append(TransferFunction([1], [1, -pole]))
            F_gain /= np.abs(pole)

    return F, F_gain

def factor_zeros(zeros):
    """
    Factor zeros into sub-TF models.
    """
    F = []
    F_gain = 1

    for zero in zeros:
        if np.iscomplex(zero):
            F.append(TransferFunction([1, -2 * np.real(zero), np.abs(zero)**2 + np.imag(zero)**2], [1]))
            F_gain *= np.abs(zero)**2
        else:
            F.append(TransferFunction([1], [1, -zero]))
            F_gain *= np.abs(zero)

    return F, F_gain

def tf_factor(sys):
    """
    Factor a transfer function into constant, real pole/zero, and 
    conjugate pole/zero pair sub-TF models.
    """
    if not isinstance(sys, TransferFunction):
        raise ValueError("Input 'sys' must be a TransferFunction object.")

    poles, zeros = extract_poles_and_zeros(sys)
    F_poles, F_poles_gain = factor_poles(poles)
    F_zeros, F_zeros_gain = factor_zeros(zeros)

    F = F_poles + F_zeros + [TransferFunction([F_poles_gain * F_zeros_gain], [1])]

    # Check by concatenation
    tf_composite = TransferFunction([1], [1])
    for sub_tf in F:
        tf_composite *= sub_tf

    print(tf_composite)
    
    num_gain = sys.num[0][0][0]
    den_gain = sys.den[0][0][0]

    if np.allclose(num_gain / den_gain, F_poles_gain * F_zeros_gain):
        return F
    else:
        raise ValueError("Composite check failed!")

# Example usage:
if __name__ == "__main__":
    sys = TransferFunction([-0.64, -0.4101, 0.00783], [1, 1.489, 0.7681, 0.09455, 0.0424, 0.7])
    tf_factors = tf_factor(sys)
    print(tf_factors)
