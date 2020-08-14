from pytest_check import check
import numpy as np

import fenics
import fenics_adjoint as fa
import ufl

import theano

theano.config.optimizer = "fast_compile"
theano.config.compute_test_value = "ignore"

from fenics_pymc3 import create_fenics_theano_op
from fenics_pymc3 import FenicsVJPOp

from fenics_numpy import evaluate_primal, evaluate_vjp


mesh = fa.UnitSquareMesh(3, 2)
V = fenics.FunctionSpace(mesh, "P", 1)


def assemble_fenics(u, kappa0, kappa1):

    f = fa.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    J = fa.assemble(J_form)
    return J


templates = (fa.Function(V), fa.Constant(0.0), fa.Constant(0.0))
inputs = (np.ones(V.dim()), np.ones(1) * 0.5, np.ones(1) * 0.6)


def test_theano_primal():
    theano.config.compute_test_value = "ignore"
    hh = create_fenics_theano_op(templates)(assemble_fenics)
    x = theano.tensor.vector()
    y = theano.tensor.vector()
    z = theano.tensor.vector()
    f = theano.function([x, y, z], hh(x, y, z))
    theano_output = f(*inputs)
    numpy_putput = evaluate_primal(assemble_fenics, templates, *inputs)[0]
    assert np.isclose(theano_output, numpy_putput)


def test_theano_vjp():
    theano.config.compute_test_value = "ignore"
    numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
        assemble_fenics, templates, *inputs
    )
    vjp_op = FenicsVJPOp(
        assemble_fenics, templates, fenics_output, tuple(fenics_inputs), tape
    )
    g = theano.tensor.vector()
    f = theano.function([g], vjp_op(g))
    theano_output = f(np.ones(1))

    numpy_output = evaluate_vjp(np.ones(1), fenics_output, tuple(fenics_inputs), tape)
    for to, no in zip(theano_output, numpy_output):
        with check:
            assert np.allclose(to, no)
