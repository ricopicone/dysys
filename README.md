# sysdynutils
A Python package for system dynamics and control systems utilities (using numpy, sympy, and control)

# Installation

This package is **not** yet published on PyPI and cannot be installed with `pip` or `conda`. For now, install the package with the following steps:

Obtain a copy of this repository and place it (now a folder on your machine) in the same directory as the code you'd like to import it into (or on your Python path). There are a few ways to get a copy of this repository:
   1. If you're familiar with Git: `clone` or `fork` and `clone` this repository.
   2. Download this repository by clicking the `Code` button above and selecting `Download ZIP`. **Unzip** the file to create a folder called `sysdynutils-main`. Rename it to `sysdynutils`. Place it in the same directory as the code into which you'd like to import it.

# Usage

To import the package into a script, use

```python
import sysdynutils.sysdynutils as sd
```

This looks for the package in the `sysdynutils` directory of the working directory (generally the location of your script).

For further usage, see the documentation in the code. A call to the function `eigenvalue_matrix_np2sp()` would look like

```python
modal_matrix = sd.eigenvalue_matrix_np2sp(modal_matrix_) # modal_matrix_ is the numpy version
```

# Issues

If you have issues, please report them on the [issues page](https://github.com/ricopicone/sysdynutils/issues).
