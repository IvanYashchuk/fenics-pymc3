import pytest

import numpy as np

import firedrake
import firedrake_adjoint
import ufl

import pymc as pm
import theano.tensor as tt

import fdm

from fenics_pymc import create_fenics_theano_op
from fenics_pymc import to_numpy

n = 25
mesh = firedrake.UnitSquareMesh(n, n)
V = firedrake.FunctionSpace(mesh, "P", 1)
DG = firedrake.FunctionSpace(mesh, "DG", 0)


def solve_firedrake(kappa0, kappa1):

    x = firedrake.SpatialCoordinate(mesh)
    f = x[0]

    u = firedrake.Function(V)
    bcs = [firedrake.DirichletBC(V, firedrake.Constant(0.0), "on_boundary")]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    JJ = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    v = firedrake.TestFunction(V)
    F = firedrake.derivative(JJ, u, v)
    firedrake.solve(F == 0, u, bcs=bcs)
    return u


templates = (firedrake.Constant(0.0), firedrake.Constant(0.0))
# inputs = (np.ones(1) * 0.5, np.ones(1) * 0.6)

# templates = (firedrake.Function(DG), firedrake.Function(DG))

true_kappa0 = firedrake.Constant(1.25)
true_kappa1 = firedrake.Constant(0.55)

# true_kappa0 = firedrake.Function(DG)
# true_kappa0.interpolate(firedrake.Constant(1.25))
# true_kappa1 = firedrake.Function(DG)
# true_kappa1.interpolate(firedrake.Constant(0.55))

true_solution = solve_firedrake(true_kappa0, true_kappa1)
true_solution_numpy = to_numpy(true_solution)

# perturb state solution and create synthetic measurements
noise_level = 0.05
MAX = np.linalg.norm(true_solution_numpy)
noise = np.random.normal(scale=noise_level * MAX, size=true_solution_numpy.size)
noisy_solution = true_solution_numpy + noise

theano_fem = create_fenics_theano_op(templates)(solve_firedrake)

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
