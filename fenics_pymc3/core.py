import theano
import theano.tensor as tt
import theano.tests.unittest_tools

import fenics
import fenics_adjoint
import pyadjoint
import ufl

import numpy as np

import functools
import itertools

from .helpers import (
    numpy_to_fenics,
    fenics_to_numpy,
    get_numpy_input_templates,
    check_input,
    convert_all_to_fenics,
)
from .helpers import FenicsVariable

from typing import Type, List, Union, Iterable, Callable, Tuple


def fem_eval(
    fenics_function: Callable,
    fenics_templates: Iterable[FenicsVariable],
    *args: np.array,
) -> Tuple[np.array, FenicsVariable, Tuple[FenicsVariable], pyadjoint.Tape]:
    """Computes the output of a fenics_function and saves a corresponding gradient tape
    Input:
        fenics_function (callable): FEniCS function to be executed during the forward pass
        fenics_templates (iterable of FenicsVariable): Templates for converting arrays to FEniCS types
        args (tuple): NumPy array representation of the input to fenics_function
    Output:
        numpy_output (np.array): NumPy array representation of the output from fenics_function(*fenics_inputs)
        residual_form (ufl.Form): UFL Form for the residual used to solve the problem with fenics.solve(F==0, ...)
        fenics_inputs (list of FenicsVariable): FEniCS representation of the input args
    """

    check_input(fenics_templates, *args)
    fenics_inputs = convert_all_to_fenics(fenics_templates, *args)

    # Create tape associated with this forward pass
    tape = pyadjoint.Tape()
    pyadjoint.set_working_tape(tape)
    fenics_output = fenics_function(*fenics_inputs)

    if isinstance(fenics_output, tuple):
        raise ValueError("Only single output from FEniCS function is supported.")

    numpy_output = np.asarray(fenics_to_numpy(fenics_output))
    return numpy_output, fenics_output, fenics_inputs, tape


def vjp_fem_eval_impl(
    g: np.array,
    fenics_output: FenicsVariable,
    fenics_inputs: Iterable[FenicsVariable],
    tape: pyadjoint.Tape,
) -> Tuple[np.array]:
    """Computes the gradients of the output with respect to the inputs."""
    # Convert tangent covector (adjoint) to a FEniCS variable
    adj_value = numpy_to_fenics(g, fenics_output)
    if isinstance(adj_value, (fenics.Function, fenics_adjoint.Function)):
        adj_value = adj_value.vector()

    tape.reset_variables()
    fenics_output.block_variable.adj_value = adj_value
    with tape.marked_nodes(fenics_inputs):
        tape.evaluate_adj(markings=True)
    fenics_grads = [fi.block_variable.adj_value for fi in fenics_inputs]

    # Convert FEniCS gradients to numpy array representation
    numpy_grads = (np.asarray(fenics_to_numpy(fg)) for fg in fenics_grads)

    return numpy_grads


# class FenicsOp(tt.Op):
#     itypes = [tt.dscalar]
#     otypes = [tt.dscalar]

#     def perform(self, node, inputs, outputs):
#         (theta,) = inputs
#         mu = mu_from_theta(theta)
#         outputs[0][0] = np.array(mu)

#     def grad(self, inputs, g):
#         (theta,) = inputs
#         mu = self(theta)
#         thetamu = theta * mu
#         return [-g[0] * mu ** 2 / (1 + thetamu + tt.exp(-thetamu))]
