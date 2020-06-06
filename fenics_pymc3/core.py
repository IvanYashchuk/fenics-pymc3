import theano
from theano.gof import Op, Apply

import fenics
import fenics_adjoint
import pyadjoint

import numpy as np

import functools

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
    numpy_grads = (
        None if fg is None else np.asarray(fenics_to_numpy(fg)) for fg in fenics_grads
    )

    return numpy_grads


class FenicsOp(Op):
    __props__ = ("ofunc", "templates", "tape")

    def __init__(self, ofunc, templates):
        self.ofunc = ofunc
        self.templates = templates
        self.tape = None
        self.fenics_output = None
        self.fenics_input = None

    def make_node(self, *inputs):
        n_inputs = len(self.templates)
        assert n_inputs == len(inputs)
        return Apply(
            self,
            [theano.tensor.as_tensor_variable(x) for x in inputs],
            [theano.tensor.dvector],
        )

    def perform(self, node, inputs, outputs):
        numpy_output, fenics_output, fenics_inputs, tape = fem_eval(
            self.ofunc, self.templates, *inputs
        )
        self.tape = tape
        self.fenics_output = fenics_output
        self.fenics_input = fenics_inputs
        outputs[0][0] = numpy_output

    def grad(self, inputs, output_grads):
        if self.tape is None or self.fenics_input is None or self.fenics_output is None:
            raise AttributeError(
                "Something went wrong during the forward pass and tape is not saved"
            )
        numpy_grads = vjp_fem_eval_impl(
            output_grads, self.fenics_output, self.fenics_inputs, self.tape
        )

        theano_grads = [
            theano.gradient.grad_undefined(self, i, inputs[i])
            if ng is None
            else theano.shared(ng)
            for i, ng in enumerate(numpy_grads)
        ]
        return theano_grads


def create_fenics_theano_op(fenics_templates: FenicsVariable) -> Callable:
    """Return `f(*args) = build_jax_fem_eval(*args)(ofunc(*args))`.
    Given the FEniCS-side function ofunc(*args), return the Theano Op,
    that is callable and differentiable in Theano programs,
    `f(*args) = create_fenics_theano_op(*args)(ofunc(*args))` with
    the VJP of `f`, where:
    `*args` are all arguments to `ofunc`.
    Args:
    ofunc: The FEniCS-side function to be wrapped.
    Returns:
    `f(args) = create_fenics_theano_op(*args)(ofunc(*args))`
    """

    def decorator(fenics_function: Callable) -> Callable:

        theano_op = FenicsOp(fenics_function, fenics_templates)

        @functools.wraps(fenics_function)
        def jax_fem_eval(*args):
            return theano_op(*args)

        return jax_fem_eval

    return decorator
