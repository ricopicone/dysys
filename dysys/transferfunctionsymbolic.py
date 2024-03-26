import sympy as sp
import numpy as np
import control

class TransferFunctionSymbolic:
    """Represents a SISO continuous LTI transfer function model in symbolic form"""

    def __init__(self, H, s=None):
        if s is not None:
            self.s = s
        else:
            self.s = sp.symbols("s", complex=True)
            # self.s = list(H.free_symbols)[0]  # Don't need this because sp.symbols("s") is unique
        num, den = sp.fraction(H)
        self.num = num.collect(self.s)
        self.den = den.collect(self.s)
        self.H = self.num/self.den
        sp.init_printing(order='grevlex')
        
    def __call__(self, s):
        """Evaluate the transfer function at a complex frequency s"""
        return self.H.subs(self.s, s)
    
    def __str__(self):
        return sp.pretty(self.H)
    
    def __factor_p(self, p, poles=True):
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
        num, den = sp.fraction(self.H.cancel())
        Kz, factors_z = self.__factor_p(num, poles=False)
        Kp, factors_p = self.__factor_p(den, poles=True)
        K = Kz * Kp
        factors = factors_p + factors_z
        return K, factors
        
    def __num_den_lists(self, params: dict = {}):
        """Returns num and den coefficients as lists"""
        num, den = self.num, self.den
        num = np.array(sp.Poly(num.evalf(subs=params), self.s).all_coeffs())
        den = np.array(sp.Poly(den.evalf(subs=params), self.s).all_coeffs())
        num = num.astype(float)
        den = den.astype(float)
        return num, den

    def to_control(self, params: dict = {}):
        """Returns an equivalent Control Systems package control.TransferFunction object"""
        num, den = self.__num_den_lists(params=params)
        return control.tf(num, den)
    
    def poles(self):
        """Returns a dict of the symbolic poles as keys and multiplicity as values"""
        return sp.roots(self.den.as_poly(self.s), strict=True)
    
    def zeros(self):
        """Returns a dict of the symbolic zeros as keys and multiplicity as values"""
        return sp.roots(self.num.as_poly(self.s), strict=True)
    
    def dc_gain(self):
        """Returns the DC gain of the transfer function"""
        return self.__call__(0).simplify()
    
    def frequency_response_function(self, w: sp.Symbol = sp.symbols("w", real=True)):
        """Returns the symbolic frequency response function (FRF)
        
        Evaluates H(jw). Assumes the region of convergence for the corresponding 
        Fourier transform is congruent with that of the Laplace transform.
        
        Args:
            w: The frequency symbol
        """
        return self.__call__(1j * w)
    
    def forced_response(
            self, 
            t: sp.Symbol, 
            u: sp.Expr = None, 
            U: sp.Expr = None,
            laplace: bool = False,
        ):
        """Returns the forced response of a SISO system
        
        The inverse Laplace transform is used to compute the forced response.
        Exactly one of arguments u or U may be provided.
        
        Args:
            t: The time symbol
            u: The input as a time-dependent expression
            U: The input as a Laplace transform (must use 
                symbolic sp.symbols("s") if using this option)
            laplace: If True, returns Laplace transform Y(s) of the output
        """
        if (u is None) and (U is None):
            raise(Exception("Must provide input as u(t) or U(s)"))
        elif (u is not None) and (U is not None):
            raise(Exception("Must provide input as just one of u(t) or U(s), not both"))
        if u is not None:
            U = sp.laplace_transform(u, t, self.s, noconds=True)
        Y = (self.H * U).simplify()
        if laplace:
            return Y
        else:
            y = sp.inverse_laplace_transform(Y, self.s, t, noconds=True)
            return y.simplify()
        

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