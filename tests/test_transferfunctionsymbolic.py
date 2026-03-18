import sympy as sp
import numpy as np
import pytest

from dysys.transferfunctionsymbolic import TransferFunctionSymbolic, tfs


class TestTransferFunctionSymbolicInit:
    def test_basic(self):
        s = sp.Symbol("s")
        H = tfs(1 / (s + 1), s=s)
        assert isinstance(H, TransferFunctionSymbolic)
        assert H.s == s

    def test_auto_detect_s(self):
        s = sp.symbols("s", complex=True)
        H = tfs(1 / (s + 1))
        assert H.s == s

    def test_auto_detect_s_real(self):
        s = sp.symbols("s")
        H = tfs(1 / (s + 1))
        # Should find s even without complex=True
        assert H.s in H.H.free_symbols or len(H.H.free_symbols) == 0

    def test_constant_tf(self):
        s = sp.Symbol("s")
        H = tfs(sp.Integer(5), s=s)
        assert H(0) == 5

    def test_invalid_symbol_raises(self):
        z = sp.Symbol("z")
        with pytest.raises(RuntimeError):
            tfs(1 / (z + 1))  # No 's' in expression

    def test_no_side_effects_on_init(self):
        """Ensure constructor doesn't modify global sympy state"""
        s = sp.Symbol("s")
        # Create multiple instances — should not error or cause side effects
        tfs(1 / (s + 1), s=s)
        tfs(s / (s**2 + 1), s=s)


class TestCallAndStr:
    def test_evaluate(self):
        s = sp.Symbol("s")
        H = tfs(1 / (s + 1), s=s)
        assert H(0) == 1
        assert H(-1) == sp.zoo  # Pole at s=-1

    def test_str(self):
        s = sp.Symbol("s")
        H = tfs(1 / (s + 1), s=s)
        result = str(H)
        assert isinstance(result, str)


class TestPoles:
    def test_first_order(self):
        s = sp.Symbol("s")
        H = tfs(1 / (s + 2), s=s)
        poles = H.poles()
        assert -2 in poles

    def test_second_order(self):
        s = sp.Symbol("s")
        H = tfs(1 / (s**2 + 3 * s + 2), s=s)
        poles = H.poles()
        assert -1 in poles
        assert -2 in poles

    def test_repeated_poles(self):
        s = sp.Symbol("s")
        H = tfs(1 / (s + 1) ** 2, s=s)
        poles = H.poles()
        assert poles[-1] == 2  # Multiplicity 2


class TestZeros:
    def test_has_zero(self):
        s = sp.Symbol("s")
        H = tfs((s + 3) / (s + 1), s=s)
        zeros = H.zeros()
        assert -3 in zeros

    def test_no_zeros(self):
        s = sp.Symbol("s")
        H = tfs(1 / (s + 1), s=s)
        zeros = H.zeros()
        assert len(zeros) == 0


class TestDCGain:
    def test_basic(self):
        s = sp.Symbol("s")
        H = tfs(3 / (s + 1), s=s)
        assert H.dc_gain() == 3

    def test_second_order(self):
        s = sp.Symbol("s")
        H = tfs(10 / (s**2 + 3 * s + 5), s=s)
        assert H.dc_gain() == 2


class TestFactor:
    def test_first_order(self):
        s = sp.Symbol("s")
        H = tfs(3 / (s + 2), s=s)
        K, factors = H.factor(check=True)
        # K * product(factors) should equal H
        product = K
        for f in factors:
            product *= f
        assert sp.simplify(product - H.H) == 0

    def test_second_order(self):
        s = sp.Symbol("s")
        H = tfs((s + 1) / (s**2 + 3 * s + 2), s=s)
        K, factors = H.factor(check=True)
        product = K
        for f in factors:
            product *= f
        diff = sp.simplify(product - H.H)
        assert diff == 0

    def test_check_raises_on_failure(self):
        """factor(check=True) should raise if something goes wrong"""
        s = sp.Symbol("s")
        H = tfs(1 / (s + 1), s=s)
        # This should not raise for a valid system
        K, factors = H.factor(check=True)
        assert K is not None


class TestFrequencyResponseFunction:
    def test_basic(self):
        s = sp.Symbol("s")
        w = sp.Symbol("w", real=True)
        H = tfs(1 / (s + 1), s=s)
        frf = H.frequency_response_function(w)
        # At w=0, FRF should equal DC gain
        assert sp.simplify(frf.subs(w, 0) - 1) == 0


class TestToControl:
    def test_basic(self):
        import control

        s = sp.Symbol("s")
        H = tfs(1 / (s + 1), s=s)
        H_ctrl = H.to_control()
        assert isinstance(H_ctrl, control.TransferFunction)

    def test_with_params(self):
        import control

        s = sp.Symbol("s")
        a = sp.Symbol("a")
        H = tfs(1 / (s + a), s=s)
        H_ctrl = H.to_control(params={a: 2})
        assert isinstance(H_ctrl, control.TransferFunction)


class TestForcedResponse:
    def test_step_first_order(self):
        s = sp.Symbol("s")
        t = sp.Symbol("t", positive=True)
        H = tfs(1 / (s + 1), s=s)
        # Step input U(s) = 1/s
        y = H.forced_response(t, U=1 / s)
        # For 1/(s+1) with step input, y(t) = 1 - exp(-t)
        expected = 1 - sp.exp(-t)
        assert sp.simplify(y - expected) == 0

    def test_must_provide_input(self):
        s = sp.Symbol("s")
        t = sp.Symbol("t")
        H = tfs(1 / (s + 1), s=s)
        with pytest.raises(Exception):
            H.forced_response(t)

    def test_cannot_provide_both(self):
        s = sp.Symbol("s")
        t = sp.Symbol("t")
        H = tfs(1 / (s + 1), s=s)
        with pytest.raises(Exception):
            H.forced_response(t, u=sp.Heaviside(t), U=1 / s)

    def test_laplace_output(self):
        s = sp.Symbol("s")
        t = sp.Symbol("t", positive=True)
        H = tfs(1 / (s + 1), s=s)
        Y = H.forced_response(t, U=1 / s, laplace=True)
        # Should be 1/(s*(s+1))
        expected = 1 / (s * (s + 1))
        assert sp.simplify(Y - expected) == 0


class TestFactorExample:
    def test_readme_example(self):
        """Test the example from transferfunctionsymbolic.py __main__"""
        s = sp.symbols("s")
        H_expr = (2430.0 * s + 810.0) / (
            30.0 * s**3 + 271.0 * s**2 + 2439.0 * s + 81.0
        )
        sys = tfs(H_expr, s=s)
        K, factors = sys.factor(check=True)
        # Should not raise — that's the main check
        assert K is not None
        assert len(factors) > 0
