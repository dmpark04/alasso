from alasso.regularizers import asymmetric_plus_abs_power_function, get_asymmetric_plus_abs_power_regularizer
import tensorflow as tf
import numpy as np

"""
A protocol is a function that takes as input some parameters and returns a tuple:
    (protocol_name, optimizer_kwargs)
The protocol name is just a string that describes the protocol.
The optimizer_kwargs is a dictionary that will get passed to KOOptimizer. It typically contains:
    step_updates, task_updates, task_metrics, regularizer_fn
"""

ALASSO_PROTOCOL = lambda a_param, a_prime, epsilon, epsilon_prime, omega_smoothing, xi: (
        'ALASSO[as=%s^%s,epsilons=%s^%s,omega_smoothing=%s,xi=%s]'%(a_param,a_prime,epsilon, epsilon_prime,omega_smoothing,xi),
{
    'init_updates':  [
        ('cweights', lambda vars, w, prev_val: w.value() ),
        ],
    'step_updates':  [
        ('grads2', lambda vars, w, prev_val: prev_val -vars['unreg_grads'][w] * vars['deltas'][w] ),
        ],
    'task_updates':  [
        ('omega',     lambda vars, w, prev_val: (tf.where(
                                                     w < vars['cweights'][w], 
                                                     tf.fill(prev_val.get_shape(), -1.0), 
                                                     tf.ones_like(prev_val)
                                                )) * tf.maximum( 
                                                         omega_smoothing * (vars['grads2'][w] - vars['oopt'].c_prime * asymmetric_plus_abs_power_function(2.0, a_prime, epsilon_prime, prev_val, vars['cweights'][w], w))/((tf.abs(vars['cweights'][w]-w.value()))**2.0+xi) + (1.0 - omega_smoothing) * tf.abs(prev_val), 
                                                         tf.abs(prev_val)
                                                     ) ), # Signed
        ('cweights',  lambda vars, w, prev_val: w.value()),
        ('grads2', lambda vars, w, prev_val: tf.zeros_like(prev_val)),
    ],
    'regularizer_fn': get_asymmetric_plus_abs_power_regularizer(2.0, a_param, epsilon),
})

