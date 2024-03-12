import sympy as sp

class TransferFunctionSymbolic:
    """Represents a continuous LTI transfer function model in symbolic form"""

    def __init__(self, H, s=None):
        self.H = H
        if s is not None:
            self.s = s
        else:
            self.s = list(H.free_symbols)[0]
    
    def factor_p(self, p, poles=True):
        p = sp.Poly(p, self.s).factor_list()  # Factored
        if poles:
            K = p[0]  # Overall gain
        else:
            K = 1/p[0]  # Overall gain
        factors = []
        for pi in p[1]:  # Each factor
            m = pi[1]  # Multiplicity
            d = sp.Poly(pi[0], self.s).degree()
            cs_k = sp.Poly(pi[0], self.s).all_coeffs()
            if d == 2:
                k = cs_k[0]  # factor gain
                cs = list(map(lambda c: c/k, cs_k))
                wn2 = cs[2]
                quad = self.s**2 + cs[1]*self.s + cs[2]
                if poles:
                    factor = wn2/quad
                    k = k/wn2
                else:
                    factor = quad/wn2
                    k = wn2/k
            elif d == 1:
                k = cs_k[1]
                tau = cs_k[0]/k
                lin = tau*self.s + 1
                if poles:
                    factor = lin
                else:
                    factor = 1/lin
            elif d == 0:
                factor = 1
                if poles:
                    k = cs_k[0]
                else:
                    k = 1/cs_k[0]
            else:
                raise(RuntimeError(f"Polynomial should not be degree {d}"))
            for _ in range(0, m):
                K = k*K
                factors.append(factor)
        return K, factors

    def factor(self):
        """Returns an overall gain and a list of standard-form terms"""
        num, den = sp.fraction(H.cancel())
        Kz, factors_z = self.factor_p(num, poles=False)
        Kp, factors_p = self.factor_p(den, poles=True)
        K = Kz * Kp
        factors = factors_p + factors_z
        return K, factors

    def to_control(self, params: dict = {}):
        """Returns an equivalent Control Systems package control.StateSpace object"""
        A, B, C, D = self.to_numpy(params=params)
        return control.ss(A, B, C, D)


def tfs(H, s=None):
    """Create a TransferFunctionSymbolic object"""
    return TransferFunctionSymbolic(H, s=s)

# Example usage:
if __name__ == "__main__":
    s = sp.symbols("s")
    H = (2430.0*s + 810.0)/(30.0*s**3 + 271.0*s**2 + 2439.0*s + 81.0)
    sys = tfs(H)
    K, factors = sys.factor()
    print(f"Overall gain: {K}")
    print(f"Factors: {factors}")