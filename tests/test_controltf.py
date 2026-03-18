import numpy as np
import control
import pytest

from dysys.controltf import TransferFunction, tf


class TestTransferFunctionCreation:
    def test_basic(self):
        H = TransferFunction([1], [1, 1])
        assert H.ninputs == 1
        assert H.noutputs == 1

    def test_tf_factory(self):
        H = tf([1], [1, 1])
        assert isinstance(H, TransferFunction)


class TestPopConjugate:
    def test_finds_conjugate(self):
        H = TransferFunction([1], [1, 1])
        root = -1 + 2j
        roots = [-1 - 2j, -3.0]
        conj = H.pop_conjugate(root, roots)
        assert np.isclose(conj, -1 - 2j)
        assert len(roots) == 1


class TestPolyFactorsCanonical:
    def test_real_roots(self):
        H = TransferFunction([1], [1, 1])
        # Polynomial (s+2)(s+5) = s^2 + 7s + 10
        factors = H.poly_factors_canonical([1, 7, 10])
        # Should have gain + 2 factors
        assert len(factors) == 3  # gain + two first-order

    def test_complex_roots(self):
        H = TransferFunction([1], [1, 1])
        # s^2 + 2s + 5 has roots -1 +/- 2j
        factors = H.poly_factors_canonical([1, 2, 5])
        # gain + one second-order factor
        assert len(factors) == 2


class TestFactorCanonical:
    def test_first_order(self):
        H = TransferFunction([3], [1, 1])
        factors = H.factor_canonical(check=True)
        # Product of factors should equal original
        product = 1
        for f in factors:
            product *= f
        product = product.minreal()
        np.testing.assert_allclose(
            np.array(product.num).flatten() / np.array(product.den).flatten()[0],
            np.array(H.num).flatten().astype(float)
            / np.array(H.den).flatten().astype(float)[0],
            rtol=1e-6,
        )

    def test_second_order(self):
        # s^2 + 2s + 5 in denominator
        H = TransferFunction([1], [1, 2, 5])
        factors = H.factor_canonical(check=True)
        assert len(factors) >= 2  # gain + at least one factor

    def test_mimo_raises(self):
        # Create a MIMO-like system via concatenation
        H = TransferFunction([1], [1, 1])
        H_mimo = control.append(H, H)
        # Wrap as our subclass manually
        H_sub = TransferFunction.__new__(TransferFunction)
        H_sub.__dict__.update(H_mimo.__dict__)
        with pytest.raises(NotImplementedError):
            H_sub.factor_canonical()

    def test_complex_system(self):
        """Test the example from the README"""
        H = TransferFunction(
            [1_000_000, 300_000_000],
            [1, 1030, 40200, 10_300_000, 100_000_000],
        )
        factors = H.factor_canonical(check=True)
        assert len(factors) >= 2
