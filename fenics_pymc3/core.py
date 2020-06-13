import theano
from theano.gof import Op, Apply
from theano.gradient import DisconnectedType

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


class FenicsVJPOp(Op):
    __props__ = ("ofunc", "templates")

    def __init__(self, ofunc, templates):
        self.ofunc = ofunc
        self.templates = templates

    def make_node(self, *inputs):
        # fenics_output = theano.gof.generic()
        # fenics_inputs = theano.gof.generic()
        # tape = theano.gof.generic()
        return Apply(
            self,
            # [theano.tensor.as_tensor_variable(inputs[0]), fenics_output, fenics_inputs, tape],
            [theano.tensor.as_tensor_variable(x) for x in inputs],
            [theano.tensor.dvector() for x in self.templates],
        )

    def perform(self, node, inputs, outputs):

        # g, fenics_output, fenics_inputs, tape = inputs
        g = inputs[0]
        primal_inputs = inputs[1:]
        numpy_output, fenics_output, fenics_inputs, tape = fem_eval(
            self.ofunc, self.templates, *primal_inputs
        )
        numpy_grads = vjp_fem_eval_impl(
            g, fenics_output, fenics_inputs, tape
        )

        # theano_grads = [
        #     theano.gradient.grad_undefined(self, i, inputs[i])
        #     if ng is None
        #     else theano.shared(ng)
        #     for i, ng in enumerate(list(numpy_grads))
        # ]

        theano_grads = [
            ng
            for i, ng in enumerate(list(numpy_grads))
        ]

        for i, tg in enumerate(theano_grads):
            outputs[i][0] = tg


class FenicsOp(Op):
    __props__ = ("ofunc", "templates")
    default_output = 0

    def __init__(self, ofunc, templates):
        self.ofunc = ofunc
        self.templates = templates
        # self.vjp_op = vjp_op

    def make_node(self, *inputs):
        n_inputs = len(self.templates)
        assert n_inputs == len(inputs)
        fenics_output = theano.gof.generic()
        fenics_inputs = theano.gof.generic()
        tape = theano.gof.generic()
        return Apply(
            self,
            [theano.tensor.as_tensor_variable(x) for x in inputs],
            # [theano.tensor.dvector(), fenics_output, fenics_inputs, tape],
            [theano.tensor.dvector()],
        )

    def perform(self, node, inputs, outputs):
        numpy_output, fenics_output, fenics_inputs, tape = fem_eval(
            self.ofunc, self.templates, *inputs
        )
        outputs[0][0] = numpy_output
        # outputs[1][0] = fenics_output
        # outputs[2][0] = fenics_inputs
        # outputs[3][0] = tape

    # def L_op(self, inputs, outputs, output_grads):
    #     # numpy_output, fenics_output, fenics_inputs, tape = outputs
    #     # g_numpy_output, _, _, _ = output_grads
    #     # if tape is None or fenics_inputs is None or fenics_output is None:
    #     #     raise AttributeError(
    #     #         "Something went wrong during the forward pass and tape is not saved"
    #     #     )

    #     # Replace gradients wrt disconnected variables with
    #     # zeros. This is a work-around for issue #1063.
    #     # copied from
    #     # https://github.com/Theano/Theano/blob/master/theano/tensor/nlinalg.py#L350
    #     # g_outputs = _zero_disconnected([numpy_output], [g_numpy_output])

    #     # vjp_op = FenicsVJPOp(self.ofunc, self.templates)
    #     # theano_grads = self.vjp_op(g_outputs[0], fenics_output, fenics_inputs, tape)
    #     theano_grads = self.vjp_op(output_grads[0], *inputs)
    #     return theano_grads

    def grad(self, inputs, output_grads):
        g, = output_grads
        vjp_op = FenicsVJPOp(self.ofunc, self.templates)
        theano_grads = vjp_op(g, *inputs)
        return theano_grads


def _zero_disconnected(outputs, grads):
    l = []
    for o, g in zip(outputs, grads):
        if isinstance(g.type, DisconnectedType):
            l.append(o.zeros_like())
        else:
            l.append(g)
    return l

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

        # vjp_op = FenicsVJPOp(fenics_function, fenics_templates)
        theano_op = FenicsOp(fenics_function, fenics_templates)

        @functools.wraps(fenics_function)
        def jax_fem_eval(*args):
            return theano_op(*args)

        return jax_fem_eval

    return decorator

def create_fenics_theano_vjp_op(fenics_templates: FenicsVariable) -> Callable:

    def decorator(fenics_function: Callable) -> Callable:

        vjp_op = FenicsVJPOp(fenics_function, fenics_templates)

        @functools.wraps(fenics_function)
        def jax_fem_eval(*args):
            return vjp_op(*args)

        return jax_fem_eval

    return decorator
