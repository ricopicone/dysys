import control
from scipy.signal import tf2zpk
import numpy as np
import numpy.polynomial.polynomial as poly

class TransferFunction(control.TransferFunction):
    """Subclass of control.TransferFunction with extra methods"""
    def pop_conjugate(self, root, roots):
        """Pop the conjugate of a root in roots list
        
        Finds the closest root in roots to the complex conjugate of root.
        Uses the "closest" metric of the Euclidean distance.
        """
        rootsa = np.array(roots)
        diff = root.conjugate() - rootsa
        dist = np.abs(diff)
        imini = dist.argmin()
        mini = roots.pop(imini)
        return mini
    
    def poly_factors_canonical(self, p):
        """Returns polynomial factors in canonical form"""
        p = poly.Polynomial(np.flip(p))  # Polynomials have increasing powers
        n = p.degree()
        roots = list(p.roots())
        K = p.coef[-1]
        factors = []
        while roots:
            root = roots.pop()
            if np.iscomplex(root):
                root_conj = self.pop_conjugate(root, roots)
                factor = poly.polyfromroots([root, root_conj])  # $s^2 + 2\zeta\omega_n s + \omega_n^2$
                factor = np.flip(np.real(factor))  # Should all be real
                k = 1/factor[-1]  # Factor gain $1/w_n^2$
                K = K/k  # Overall gain absorbing factor gain
            else:
                factor = poly.polyfromroots([root])
                factor = np.flip(np.real(factor))  # $s + 1/\tau$
                print(factor)
                K = K*factor[-1]  # Overall gain absorbing $\tau$
                factor = factor/factor[-1]  # $\tau s + 1$
                k = 1  # Factor gain
            factors.append((k, factor))
        factors.insert(0, (K, np.array([1])))  ## Prepend the gain as the first factor
        return factors
        
    def factor_canonical(self, check=False):
        """Returns a list of transfer functions in canonical form, the product of which equals self."""
        if self.ninputs > 1 or self.noutputs > 1:
            raise NotImplementedError(
                "TransferFunction.factor_canonical is currently only implemented "
                "for SISO systems.")
        zpf = self.poly_factors_canonical(np.array(self.num).flatten())  # Zero factors
        ppf = self.poly_factors_canonical(np.array(self.den).flatten())  # Pole factors
        gain = zpf.pop(0)[0]/ppf.pop(0)[0]
        tf_factors = [control.TransferFunction([gain],[1])]
        for zf in zpf:
            tf_factors.append(
                control.TransferFunction(zf[1], [1/zf[0]])
            )
        for pf in ppf:
            tf_factors.append(
                control.TransferFunction([1/pf[0]], pf[1])
            )
        if check:
            tf_from_factors = 1
            for tf in tf_factors:
                tf_from_factors *= tf
            tf_from_factors = tf_from_factors.minreal()
            print("Is", self, "equal to", tf_from_factors, "?")
            self_den_0 = np.array(self.den).flatten().astype(np.float64)[0]
            assert(
                np.allclose(
                    np.array(tf_from_factors.num).flatten(), 
                    np.array(self.num).flatten().astype(np.float64)/self_den_0
                )
            )
            assert(
                np.allclose(
                    np.array(tf_from_factors.den).flatten(),
                    np.array(self.den).flatten().astype(np.float64)/self_den_0
                )
            )
            print("Yep")
        return tf_factors
        
def tf(*args, **kwargs):
    """Create a TransferFunction object"""
    return TransferFunction(*args, **kwargs)    
        