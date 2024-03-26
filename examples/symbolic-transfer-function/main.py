import dysys as ds
import control as ct
import sympy as sp

# Create a symbolic transfer function
s = sp.symbols("s", complex=True)
a, b, c = sp.symbols("a, b, c", positive=True)
H = ds.tfs((s + 2)/(a*s**2 + b*s + c))
print(H)

# Poles and zeros
poles = H.poles()
zeros = H.zeros()
print(f"poles: {poles}", f"\nzeros:{zeros}")

# Evaluating H(s), H(jw), and the DC Gain
s1 = 1 + 3j  # A complex frequency
print(f"H({s1}): {H(s1)}")
Hjw = H.frequency_response_function()
print(f"Frequency response function H(jw): {Hjw}")
print(f"DC gain: {H.dc_gain()}")

# Factor into canonical terms (for Bode plot sketching)
print("Factored H(s):", f"\n{H.factor()}")

# Forced response
t = sp.symbols("t", real=True)
u = sp.Heaviside(t)
y = H.forced_response(t, u)
print(f"Forced response (unit step response): {y}")