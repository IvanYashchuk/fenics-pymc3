import theano
from theano.gof import Op, Apply

import functools

from fenics_numpy import evaluate_primal, evaluate_vjp


class FenicsVJPOp(Op):
    params_type = theano.gof.type.Generic()
    __props__ = ("ofunc", "templates", "fenics_output", "fenics_inputs", "tape")

    def __init__(self, ofunc, templates, fenics_output, fenics_inputs, tape):
        self.ofunc = ofunc
        self.templates = templates
        self.fenics_output = fenics_output
        self.fenics_inputs = fenics_inputs
        self.tape = tape

    def get_params(self, node):
        return (self.fenics_output, self.fenics_inputs, self.tape)

    def make_node(self, *inputs):
        return Apply(
            self,
            [theano.tensor.as_tensor_variable(x) for x in inputs],
            [theano.tensor.dvector() for x in self.templates],
        )

    def perform(self, node, inputs, outputs, params):
        Δfenics_output = inputs[0]
        fenics_output, fenics_inputs, tape = params
        numpy_grads = evaluate_vjp(Δfenics_output, fenics_output, fenics_inputs, tape)

        theano_grads = (
            theano.gradient.grad_undefined(self, i, inputs[i]) if ng is None else ng
            for i, ng in enumerate(numpy_grads)
        )

        for i, tg in enumerate(numpy_grads):
            outputs[i][0] = tg


class FenicsOp(Op):
    __props__ = ("ofunc", "templates")

    def __init__(self, ofunc, templates):
        self.ofunc = ofunc
        self.templates = templates

    def make_node(self, *inputs):
        n_inputs = len(self.templates)
        assert n_inputs == len(inputs)
        return Apply(
            self,
            [theano.tensor.as_tensor_variable(x) for x in inputs],
            [theano.tensor.dvector()],
        )

    def perform(self, node, inputs, outputs):
        numpy_output, fenics_output, fenics_inputs, tape = evaluate_primal(
            self.ofunc, self.templates, *inputs
        )

        self.vjp_op = FenicsVJPOp(
            self.ofunc, self.templates, fenics_output, tuple(fenics_inputs), tape
        )
        outputs[0][0] = numpy_output

    def grad(self, inputs, output_grads):
        theano_grads = self.vjp_op(output_grads[0])
        return theano_grads


def create_fenics_theano_op(fenics_templates):
    """Return `f(*args) = create_fenics_theano_op(*args)(ofunc(*args))`.
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

    def decorator(fenics_function):

        theano_op = FenicsOp(fenics_function, fenics_templates)

        @functools.wraps(fenics_function)
        def theano_fem_eval(*args):
            return theano_op(*args)

        return theano_fem_eval

    return decorator
