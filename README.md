# sysdynutils

A Python package for system dynamics and control systems utilities (using numpy, sympy, and control)

# Installation

This package now published on PyPI and can be installed with

```python
pip install sysdynutils
```

# Usage

To import the package into a script, use

```python
import sysdynutils as sd
```

Create a symbolic state-space model:

```python
A = [[-4, -3, 0], [0, -8, 4], [0, 0, -1]]
B = [[0], [1], [0]]
C = [[0, 1, 0]]
D = [[0]]
sys = sd.sss(A, B, C, D)  # Create a symbolic state-space model
```

Now call

For further usage, see the documentation in the code.

# Issues

If you have issues, please report them on the [issues page](https://github.com/ricopicone/sysdynutils/issues).
