import pytest

import numpy as np

import fenics
import fenics_adjoint as fa
import ufl

import fdm

from fenics_pymc import create_fenics_theano_op
from fenics_pymc import to_numpy

import pymc as pm
import theano.tensor as tt

fenics.set_log_level(fenics.LogLevel.ERROR)

n = 25
mesh = fa.UnitSquareMesh(n, n)
V = fenics.FunctionSpace(mesh, "P", 1)
DG = fenics.FunctionSpace(mesh, "DG", 0)


def solve_fenics(kappa0, kappa1):

    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    u = fa.Function(V)
    bcs = [fa.DirichletBC(V, fa.Constant(0.0), "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    v = fenics.TestFunction(V)
    F = fenics.derivative(JJ, u, v)
    fa.solve(F == 0, u, bcs=bcs)
    return u


templates = (fa.Constant(0.0), fa.Constant(0.0))
# inputs = (np.ones(1) * 0.5, np.ones(1) * 0.6)

# templates = (fa.Function(DG), fa.Function(DG))

true_kappa0 = fa.Constant(1.25)
true_kappa1 = fa.Constant(0.55)

# true_kappa0 = fa.Function(DG)
# true_kappa0.interpolate(fa.Constant(1.25))
# true_kappa1 = fa.Function(DG)
# true_kappa1.interpolate(fa.Constant(0.55))

true_solution = solve_fenics(true_kappa0, true_kappa1)
true_solution_numpy = to_numpy(true_solution)

# perturb state solution and create synthetic measurements
noise_level = 0.05
MAX = np.linalg.norm(true_solution_numpy)
noise = np.random.normal(scale=noise_level * MAX, size=true_solution_numpy.size)
noisy_solution = true_solution_numpy + noise

theano_fem = create_fenics_theano_op(templates)(solve_fenics)

with pm.Model() as fit_diffusion:
    sigma = pm.InverseGamma("sigma", alpha=3.0, beta=0.5)

    kappa0 = pm.TruncatedNormal(
        "kappa0", mu=1.0, sigma=0.5, lower=1e-5, upper=2.0, shape=(1,)
    )  # truncated(Normal(1.0, 0.5), 1e-5, 2)
    kappa1 = pm.TruncatedNormal(
        "kappa1", mu=0.7, sigma=0.5, lower=1e-5, upper=2.0, shape=(1,)
    )  # truncated(Normal(0.7, 0.5), 1e-5, 2)

    predicted_solution = pm.Deterministic("pred_sol", theano_fem(kappa0, kappa1))

    d = pm.Normal("d", mu=predicted_solution, sd=sigma, observed=noisy_solution)


def test_run_sample():
    with fit_diffusion:
        pm.sample(5, tune=5, chains=1)
