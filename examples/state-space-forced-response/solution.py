# %% [markdown]
# Load Python packages
# %%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import dysys
from pprint import pprint

# %% tags=["remove_input"]
# %matplotlib inline
import engcom

# %% [markdown]
## Define the System
# Define a symbolic state-space matrix using the `dysys` package as follows:
# %%
A = [[-4, -3, 0], [0, -8, 4], [0, 0, -1]]
B = [[0], [1], [0]]
C = [[0, 1, 0]]
D = [[0]]
sys = dysys.sss(A, B, C, D)  # Create a symbolic state-space model

# %% [markdown]
## Eigenvalues and Stability
# The eigenvalue matrix `L` can be found via the eig() method:
# %%
L, M = sys.eig()
print(f"Eigenvalues: {L.diagonal().tolist()}")

# %% [markdown]
# The real parts of the eigenvalues are all negative; therefore, the system is asymptotically stable.

# %% [markdown]
## Eigenvectors and the Modal Matrix
# The eigenvectors are stored in `M`:
# %%
print(M)

# %% [markdown]
## State Transition Matrix
# The state transition matrix $\Phi(t)$ can be found as follows:
# %%
t = sp.Symbol("t", nonnegative=True)  ## Solution valid for t >= 0
Phi = sys.state_transition_matrix(t)
pprint(Phi)

# %% [markdown]
## Forced State Response
# The forced state response for $t \ge 0$ can be found as follows:
# %%
u_s = sp.Heaviside(t)
x_fo = sys.state_forced_response(t, u=u_s)
pprint(x_fo)

# %% [markdown]
## Forced Output Response
# %%
y_fo = sys.output_forced_response(t, u=u_s)
pprint(y_fo)

# %% [markdown]
## Plot the Forced Output Response
# First, convert the symbolic expression to a NumPy function:
# %%
y_fo_fun = sp.lambdify(t, y_fo, "numpy")

# %% [markdown]
# Now create arrays to plot:
# %%
t_ = np.linspace(0, 5, 101)
y_fo_ = y_fo_fun(t_).flatten()

# %% [markdown]
# Finally, plot:
# %% tags=["remove_output"]
fig, ax = plt.subplots()
ax.plot(t_, y_fo_)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Forced Output Response $y_\\text{fo}$")
plt.show()

# %% tags=["remove_input"]
import engcom

engcom.show(fig)
