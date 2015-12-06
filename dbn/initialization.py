import numpy as np


import theano


def GlorotInit(rng, param_size, name=None):
    W_bound = 4 * np.sqrt(6.0 / (param_size[0] + param_size[1]))
    W = theano.shared(np.asarray(
                      rng.uniform(low=-W_bound, high=W_bound, size=param_size),
                      dtype=theano.config.floatX), borrow=True,
                      name=name)
    return W
