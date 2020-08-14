# from .core import fem_eval, vjp_fem_eval_impl
from .core import create_fenics_theano_op, create_fenics_theano_vjp_op
from fenics_numpy import fenics_to_numpy, numpy_to_fenics
