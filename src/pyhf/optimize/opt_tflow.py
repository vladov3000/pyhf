"""Tensorflow Optimizer Backend."""
from .. import get_backend
import tensorflow as tf


def make_func(
    objective,
    data,
    pdf,
    tv,
    fixed_values_tensor,
    fixed_idx=[],
    variable_idx=[],
    do_grad=False,
):
    tensorlib, _ = get_backend()

    if do_grad:

        def func(pars):
            pars = tensorlib.astensor(pars)
            with tf.GradientTape() as tape:
                tape.watch(pars)
                constrained_pars = tv.stitch([fixed_values_tensor, pars])
                constr_nll = objective(constrained_pars, data, pdf)
            grad = tape.gradient(constr_nll, pars).values
            return constr_nll.numpy(), grad

    else:

        def func(pars):
            pars = tensorlib.astensor(pars)
            constrained_pars = tv.stitch([fixed_values_tensor, pars])
            return objective(constrained_pars, data, pdf)

    return func
