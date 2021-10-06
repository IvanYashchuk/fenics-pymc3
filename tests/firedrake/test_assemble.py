from pytest_check import check
import numpy as np

import firedrake
import firedrake_adjoint
import ufl

import theano

from fenics_pymc import create_fenics_theano_op
from fenics_pymc import FenicsVJPOp

from fecr import evaluate_primal, evaluate_pullback

theano.config.optimizer = "fast_compile"
theano.config.compute_test_value = "ignore"

mesh = firedrake.UnitSquareMesh(3, 2)
V = firedrake.FunctionSpace(mesh, "P", 1)


def assemble_firedrake(u, kappa0, kappa1):

    x = firedrake.SpatialCoordinate(mesh)
    f = x[0]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    J_form = 0.5 * inner(kappa0 * grad(u), grad(u)) * dx - kappa1 * f * u * dx
    J = firedrake.assemble(J_form)
    return J


templates = (firedrake.Function(V), firedrake.Constant(0.0), firedrake.Constant(0.0))
inputs = (np.ones(V.dim()), np.ones(1) * 0.5, np.ones(1) * 0.6)


def test_theano_primal():
    theano.config.compute_test_value = "ignore"
    hh = create_fenics_theano_op(templates)(assemble_firedrake)
    x = theano.tensor.vector()
    y = theano.tensor.vector()
    z = theano.tensor.vector()
    f = theano.function([x, y, z], hh(x, y, z))
    theano_output = f(*inputs)
    numpy_putput = evaluate_primal(assemble_firedrake, templates, *inputs)[0]
    assert np.isclose(theano_output, numpy_putput)


def test_theano_vjp():
    theano.config.compute_test_value = "ignore"
    numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
        assemble_firedrake, templates, *inputs
    )
    vjp_op = FenicsVJPOp(
        assemble_firedrake, templates, fenics_output, tuple(fenics_inputs), tape
    )
    g = theano.tensor.vector()
    f = theano.function([g], vjp_op(g))
    theano_output = f(np.ones(1))

    numpy_output = evaluate_pullback(
        fenics_output, tuple(fenics_inputs), tape, np.ones(1)
    )
    for to, no in zip(theano_output, numpy_output):
        with check:
            assert np.allclose(to, no)
