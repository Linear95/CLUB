import tensorflow as tf
import numpy as np

def vars_from_scopes(scopes):
    """
    Returns list of all variables from all listed scopes. Operates within the current scope,
    so if current scope is "scope1", then passing in ["weights", "biases"] will find
    all variables in scopes "scope1/weights" and "scope1/biases".
    """
    current_scope = tf.get_variable_scope().name
    #print(current_scope)
    if current_scope != '':
        scopes = [current_scope + '/' + scope for scope in scopes]
    var = []
    for scope in scopes:
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            var.append(v)
    return var

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * ((x - mu) / tf.exp(log_sigma)) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)