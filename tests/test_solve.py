import pytest

import numpy as np

import fenics
import fenics_adjoint as fa
import ufl

import fdm

# from fenics_pymc3 import fem_eval, vjp_fem_eval_impl
from fenics_pymc3 import fenics_to_numpy, numpy_to_fenics

mesh = fa.UnitSquareMesh(6, 5)
V = fenics.FunctionSpace(mesh, "P", 1)


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
inputs = (np.ones(1) * 0.5, np.ones(1) * 0.6)


# def test_fenics_forward():
#     numpy_output, _, _, _ = fem_eval(solve_fenics, templates, *inputs)
#     u = solve_fenics(fa.Constant(0.5), fa.Constant(0.6))
#     assert np.allclose(numpy_output, fenics_to_numpy(u))


# def test_fenics_vjp():
#     numpy_output, fenics_output, fenics_inputs, tape = fem_eval(
#         solve_fenics, templates, *inputs
#     )
#     g = np.ones_like(numpy_output)
#     vjp_out = vjp_fem_eval_impl(g, fenics_output, fenics_inputs, tape)
#     check1 = np.isclose(vjp_out[0], np.asarray(-2.91792642))
#     check2 = np.isclose(vjp_out[1], np.asarray(2.43160535))
#     assert check1 and check2
