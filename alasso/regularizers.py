import tensorflow as tf

def quadratic_regularizer(weights, vars, norm=2):
    """Compute the regularization term.

    Args:
        weights: list of Variables
        _vars: dict from variable name to dictionary containing the variables.
            Each set of variables is stored as a dictionary mapping from weights to variables.
            For example, vars['grads'][w] would retreive the 'grads' variable for weight w
        norm: power for the norm of the (weights - consolidated weight)

    Returns:
        scalar Tensor regularization term
    """
    reg = 0.0
    for w in weights:
        reg += tf.reduce_sum(vars['omega'][w] * (w - vars['cweights'][w])**norm)
    return reg

def asymmetric_plus_abs_power_function(power, a_param, epslion, omega_with_direction, center, w):
     return tf.where(
                omega_with_direction > 0.0, 
                tf.where(
                    w < center, 
                    omega_with_direction * (tf.abs(w - center))**power, 
                    (omega_with_direction * a_param + epslion) * (tf.abs(w - center))**power
                ), 
                tf.where(
                    w < center, 
                    (-1.0 * omega_with_direction * a_param + epslion) * (tf.abs(w - center))**power, 
                    -1.0 * omega_with_direction * (tf.abs(w - center))**power
                )
            )

def get_asymmetric_plus_abs_power_regularizer(power, a_param, epslion):
    """Abs asymmetric power regularizers with different norms"""
    def _regularizer_fn(weights, vars):
        reg = 0.0
        for w in weights:
            reg += tf.reduce_sum( asymmetric_plus_abs_power_function(power, a_param, epslion, vars['omega'][w], vars['cweights'][w], w) )
        return reg
    return _regularizer_fn

