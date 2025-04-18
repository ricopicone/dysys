# %% [markdown]
# Load Python packages
# %%
import dysys
import control
import matplotlib.pyplot as plt

# %% tags=["remove_input"]
# %matplotlib inline
import engcom

# %% [markdown]
## Define the System
# Define a transfer function using the `dysys` package as follows:
# %%
H = dysys.tf(
	[1_000_000, 300_000_000],  # Numerator coef's
	[1, 1030, 40200, 10_300_000, 100_000_000]  # Denominator coef's
)

# %% [markdown]
## Factoring the Transfer Function
# Get a list of transfer functions that are the canonical factors of `H`:
# %%
factors = H.factor_canonical(check=True)  # Check that the factors are correct

# %% [markdown]
# Print the factors:
# %%
for factor in list(map(lambda x: x._repr_latex_()[1:-1], factors)):
	print(factor)

# %% [markdown]
# Generate a Bode plot:
# %%
control.bode_plot([H] + factors, dB=True, wrap_phase=True)
plt.gcf().legend(
	list(map(lambda x: x._repr_latex_()[1:-1], [H] + factors)), 
	loc="outside center right",
)
# plt.savefig("bode.svg")
plt.show()