# fenics-pymc3 &middot; [![Build](https://github.com/ivanyashchuk/fenics-pymc3/workflows/CI/badge.svg)](https://github.com/ivanyashchuk/fenics-pymc3/actions?query=workflow%3ACI+branch%3Amaster) [![Coverage Status](https://coveralls.io/repos/github/IvanYashchuk/fenics-pymc3/badge.svg?branch=master)](https://coveralls.io/github/IvanYashchuk/fenics-pymc3?branch=master) [![DOI](https://zenodo.org/badge/269920875.svg)](https://zenodo.org/badge/latestdoi/269920875)

This package enables use of [FEniCS](http://fenicsproject.org) for solving differentiable variational problems in [PyMC3](https://docs.pymc.io/).

Automatic adjoint solvers for FEniCS programs are generated with [dolfin-adjoint/pyadjoint](http://www.dolfin-adjoint.org/en/latest/).
These solvers make it possible to use Theano's (PyMC3 backend) reverse mode automatic differentiation with FEniCS.

Current limitations:
* Differentiation wrt Dirichlet boundary conditions and mesh coordinates is not implemented yet.

## Example
Here is the demonstration of fitting coefficients of a variant of the [Poisson's PDE](https://en.wikipedia.org/wiki/Poisson%27s_equation)
using PyMC3's NUTS sampler.

```python
import numpy as np
import fenics
fenics.set_log_level(fenics.LogLevel.ERROR)
import fenics_adjoint as fa
import ufl

from fenics_pymc3 import create_fenics_theano_op
from fenics_pymc3 import fenics_to_numpy, numpy_to_fenics

# Create mesh for the unit square domain
n = 10
mesh = fa.UnitSquareMesh(n, n)

# Define discrete function spaces and functions
V = fenics.FunctionSpace(mesh, "CG", 1)
W = fenics.FunctionSpace(mesh, "DG", 0)

def solve_fenics(kappa0, kappa1):
    # This function inside should be traceable by fenics_adjoint
    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    u = fa.Function(V)
    bcs = [fa.DirichletBC(V, fa.Constant(0.0), "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    v = fenics.TestFunction(V)
    F = inner(kappa0*grad(u), grad(v)) * dx - kappa1 * f * v * dx
    fa.solve(F == 0, u, bcs=bcs)
    return u

# Let's generate artificial data
true_kappa0 = fa.Constant(1.25)
true_kappa1 = fa.Constant(0.55)

true_solution = solve_fenics(true_kappa0, true_kappa1)
true_solution_numpy = fenics_to_numpy(true_solution)

# Perturb the solution with noise
noise_level = 0.05
noise = np.random.normal(scale=noise_level * np.linalg.norm(true_solution_numpy), size=true_solution_numpy.size)
noisy_solution = true_solution_numpy + noise

# Define FEniCS template representation of Theano/NumPy input
templates = (fa.Constant(0.0), fa.Constant(0.0))

# Now let's create Theano wrapper of `solve_fenics` function
theano_fem_solver = create_fenics_theano_op(templates)(solve_fenics)

# `theano_fem_solver` can now be used inside PyMC3's model
import pymc3 as pm
import theano.tensor as tt

with pm.Model() as fit_poisson:
    sigma = pm.InverseGamma("sigma", alpha=3.0, beta=0.5)

    kappa0 = pm.TruncatedNormal(
        "kappa0", mu=1.0, sigma=0.5, lower=1e-5, upper=2.0, shape=(1,)
    )
    kappa1 = pm.TruncatedNormal(
        "kappa1", mu=0.7, sigma=0.5, lower=1e-5, upper=2.0, shape=(1,)
    )
    predicted_solution = pm.Deterministic("pred_sol", theano_fem_solver(kappa0, kappa1))

    d = pm.Normal("d", mu=predicted_solution, sd=sigma, observed=noisy_solution)

with fit_poisson:
    trace = pm.sample(500, chains=4, cores=4)

pm.summary(trace)
#                 mean     sd  hdi_3%  hdi_97%  ...  ess_sd  ess_bulk  ess_tail  r_hat
# sigma          0.015  0.001   0.013    0.017  ...   689.0     715.0     723.0   1.00
# kappa0[0]      1.247  0.377   0.586    1.926  ...   334.0     331.0     462.0   1.02
# kappa1[0]      0.586  0.179   0.267    0.900  ...   352.0     352.0     582.0   1.02
```

## Installation
First install [FEniCS](http://fenicsproject.org).
Then install [dolfin-adjoint](http://www.dolfin-adjoint.org/en/latest/) with:

    python -m pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@master

Then install [numpy-fenics-adjoint](https://github.com/IvanYashchuk/numpy-fenics-adjoint) with:

    python -m pip install git+https://github.com/IvanYashchuk/numpy-fenics-adjoint@master

Then install [PyMC3](https://docs.pymc.io/) with:

    python -m pip install pymc3

After that install fenics-pymc3 with:

    python -m pip install git+https://github.com/IvanYashchuk/fenics-pymc3.git@master

## Reporting bugs

If you found a bug, create an [issue].

[issue]: https://github.com/IvanYashchuk/fenics-pymc3/issues/new

## Contributing

Pull requests are welcome from everyone.

Fork, then clone the repository:

    git clone https://github.com/IvanYashchuk/fenics-pymc3.git

Make your change. Add tests for your change. Make the tests pass:

    pytest tests/

Check the formatting with `black` and `flake8`. Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/IvanYashchuk/fenics-pymc3/pulls
