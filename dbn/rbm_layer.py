# Implementation of RBM Layer
# It can be either fully connected or convolutional

import logging
from operator import mul


import numpy as np


from neon import NervanaObject
from neon.layers.layer import ParameterLayer, Convolution
from neon.transforms import Logistic


logger = logging.getLogger(__name__)


def _calc_optree(optree, be):
    """
    Calculate operation tree and return result as Tensor
    """
    result = be.empty(optree.shape)
    result._assign(optree)
    return result


class RBMLayer(ParameterLayer):
    """
    RBM layer implementation.
    Works with volumetric data.
    Note, that some functions have extra argument "labels" but it is never used.
    It allows more uniform API

    Arguments:
        n_hidden (int): number of hidden units
        init (Optional[Initializer]): Initializer object to use for
            initializing layer weights
        name (Optional[str]): layer name. Defaults to "RBMLayer"
        persistant (bool): whether to use persistant CD
        kPCD (int): number of samples generation during negative phase of CD (CD-k)
        use_fast_weights (bool): whether to use fast weights CD algorithm for learning. Not implemented yet!
        sparse_target (float): desired sparsity
        sparse_damping (float): damping of sparsity parameters
        sparse_cost (float): cost of not matching sparsity target
        collect_zero_signal (bool): whether to use units with 0 signal during learning. Not supported.

    Note: kwargs are used only for multiple inheritance. See ConvolutionalRBMLayer
    """

    def __init__(self, n_hidden, init=None, name="RBMLayer", parallelism="Unknown",
                 persistant=False, kPCD=1, use_fast_weights=False,
                 sparse_target=0, sparse_damping=0, sparse_cost=0,
                 collect_zero_signal=True, **kwargs):
        super(RBMLayer, self).__init__(init, name, parallelism)

        self.persistant = persistant
        self.kPCD = kPCD
        self.use_fast_weights = use_fast_weights
        self.sparse_target = sparse_target
        self.sparse_damping = sparse_damping
        self.sparse_cost = sparse_cost
        self.collect_zero_signal = collect_zero_signal

        self.b_hidden = None
        self.b_visible = None

        self.sigmoid = Logistic()

        self.chain = None
        self.n_hidden = n_hidden
        self.n_visible = None

    def allocate(self, shared_outputs=None):
        super(RBMLayer, self).allocate(shared_outputs)

        # self.W must be already initialized
        self.init_params(None, self.b_vis_shape, self.b_hid_shape)

    def configure(self, in_obj):
        """
        sets shape based parameters of this layer given an input tuple or int
        or input layer

        Arguments:
            in_obj (int, tuple, Layer or Tensor or dataset): object that provides shape
                                                             information for layer

        Returns:
            (tuple): shape of output data
        """
        super(RBMLayer, self).configure(in_obj)

        #TODO: self.in_shape must be an int. Check this
        self.n_visible = self.in_shape
        if isinstance(self.in_shape, tuple):
            self.n_visible = reduce(mul, self.in_shape)

        self.out_shape = (self.n_hidden,)

        self.weight_shape = (self.n_visible, self.n_hidden)

        # bias for visible units
        self.b_vis_shape = (self.n_visible, 1)

        # bias for hidden units
        self.b_hid_shape = (self.n_hidden, 1)

        return self

    def init_params(self, shape=None, b_vis_shape=None, b_hid_shape=None):
        """
        Allocate layer parameter buffers and initialize them with the
            supplied initializer.

        Arguments:
            shape (int, tuple): shape to allocate for layer paremeter
                buffers.
        """
        # initialize self.W
        if not shape is None:
            super(RBMLayer, self).init_params(shape)

        parallel, distributed = self.get_param_attrs()
        if not b_hid_shape is None:
            self.b_hidden = self.be.zeros(b_hid_shape, parallel=parallel, distributed=distributed)
            self.db_hidden = self.be.zeros_like(self.b_hidden)

        if not b_vis_shape is None:
            self.b_visible = self.be.zeros(b_vis_shape, parallel=parallel, distributed=distributed)
            self.db_visible = self.be.zeros_like(self.b_visible)

    def get_params(self):
        """
        Get layer parameters, gradients, and states for optimization
        """
        return (((self.W, self.dW), (self.b_hidden, self.db_hidden),
                (self.b_visible, self.db_visible)), self.states)

    def get_params_serialize(self, keep_states=True):
        """
        Get layer parameters. All parameters are needed for optimization, but
        only Weights are serialized.

        Arguments:
            keep_states (bool): Control whether all parameters are returned
                or just weights for serialization. Defaults to True.
        """
        serial_dict = {'params': {'W': self.W.asnumpyarray(),
                                  'b_hidden': self.b_hidden.asnumpyarray(),
                                  'b_visible': self.b_visible.asnumpyarray(),
                                  'name': self.name}}
        if keep_states:
            serial_dict['states'] = [s.asnumpyarray() for s in self.states]
        return serial_dict

    def set_params(self, pdict):
        """
        Set layer parameters (weights). Allocate space for other parameters but do not initialize
        them.

        Arguments:
            pdict (dict): dictionary or ndarray with layer parameters
        """
        # load pdict, convert self.W to Tensor
        super(RBMLayer, self).set_params(pdict)

        self.b_hidden = self.be.array(self.b_hidden)
        self.db_hidden = self.be.empty_like(self.b_hidden)

        self.b_visible = self.be.array(self.b_visible)
        self.db_visible = self.be.array(self.b_visible)

    def fprop(self, inputs, inference=False, labels=None, weights=None):
        """
        forward propagation. Returns probability of hidden units
        """
        hidden_proba_optree = self.hidden_probability(inputs, weights=weights)
        hidden_proba = self.be.empty(hidden_proba_optree.shape)
        hidden_proba._assign(hidden_proba_optree)
        return hidden_proba

    def bprop(self, hidden_units, alpha=None, beta=None, weights=None):
        """
        CD1 backward pass (negative phase)
        Returns probability of visible units
        """
        visible_proba_optree = self.visible_probability(hidden_units, weights=weights)
        visible_proba = self.be.empty(visible_proba_optree.shape)
        visible_proba._assign(visible_proba_optree)
        return visible_proba

    def _grad(self, visible_units, hidden_units):
        """
        Calculate positive or negative gradient of weights

        Inputs:
            visible_units (Tensor): visible units
            hidden_units (Tensor): hidden_units

        Returns:
            OPTree.node
        """
        return self.be.dot(visible_units, hidden_units.T)

    def update(self, v_pos, labels=None):
        """
        Calculate gradients
        Inputs:
            v_pos (Tensor): input units (typically given input sample X) of size (n_visibles, batch_size)
            labels (Tensor): either OHE labels of shape (n_classes, batch_size) or just labels of shape (1, batch_size).
                In this case it will be converted to OHE labels.
        Returns:
            (update_W, update_b_hidden, update_b_visible) (tuple): gradients of W, b_hidden, b_visible
        """
        # positive phase
        h_pos = self.hidden_probability(v_pos)

        # negative phase
        if self.persistant:
            if self.chain is None:
                self.chain = self.be.zeros(h_pos.shape)
            chain = self.chain
        else:
            chain = h_pos > self.be.array(self.be.rng.uniform(size=h_pos.shape))

        for k in xrange(self.kPCD):
            if self.persistant:
                v_neg = self.sample_visibles(chain)
            else:
                v_neg = self.visible_probability(chain)

            h_neg = self.hidden_probability(v_neg)
            chain = h_neg > self.be.array(self.be.rng.uniform(size=h_neg.shape))

        # calculate chain explicitly
        if self.persistant:
            self.chain._assign(chain)

        if not self.collect_zero_signal:
            zero_signal_mask = self.hidden_preacts.get() == 0

            h_pos[zero_signal_mask] = self.sparse_target
            sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos)
            h_pos[zero_signal_mask] = 0
            h_neg[zero_signal_mask] = 0
        else:
            sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos)

        # update_W = self.be.dot(v_pos, h_pos.T) - self.be.dot(v_neg, h_neg.T)
        update_W = self._grad(v_pos, h_pos) - self._grad(v_neg, h_neg)
        update_b_visible = self.be.mean(v_pos - v_neg, axis=-1)
        update_b_hidden = self.be.mean(h_pos - h_neg, axis=-1) - sparsegrads_b_hidden

        result = {'W': update_W / float(self.be.bsz),
                  'b_hidden': update_b_hidden,
                  'b_visible': update_b_visible}
        if not self.collect_zero_signal:
            result['zero_signal_mask'] = zero_signal_mask
        return result

    def hidden_probability(self, inputs, labels=None, weights=None):
        """
        Calculate P(h | v)
        """
        if weights is None:
            weights = self.W

        hidden_preacts = self.be.dot(weights.T, inputs)
        hidden_proba = self.sigmoid(hidden_preacts + self.b_hidden)
        return hidden_proba

    def visible_probability(self, hidden_units, weights=None):
        """
        Calculate P(v|h)
        """
        if weights is None:
            weights = self.W

        visible_preacts = self.be.dot(weights, hidden_units) + self.b_visible
        visible_proba = self.sigmoid(visible_preacts)
        return visible_proba

    def sample_hiddens(self, visible_units, labels=None):
        """
        Sample hidden units.
        """
        h_probability = self.hidden_probability(visible_units)
        return h_probability > self.be.array(self.be.rng.uniform(size=h_probability.shape))

    def sample_visibles(self, hidden_units):
        """
        Sample visible units
        """
        v_probability = self.visible_probability(hidden_units)
        return v_probability > self.be.array(self.be.rng.uniform(size=v_probability.shape))

    def get_sparse_grads_b_hidden(self, h_proba):
        if self.sparse_cost == 0:
            return self.be.zeros_like(self.b_hidden)

        if not hasattr(self, 'hidmeans'):
            self.hidmeans = self.be.empty((self.n_hidden, 1))
            self.hidmeans[:] = self.sparse_target * self.be.ones((self.n_hidden, 1))

        hidden_probability_mean = self.be.mean(h_proba, axis=-1)
        self.hidmeans[:] = self.sparse_damping * self.hidmeans + (1 - self.sparse_damping) * hidden_probability_mean

        sparsegrads_b_hidden = self.sparse_cost * (self.hidmeans - self.sparse_target)
        return sparsegrads_b_hidden

    def free_energy(self, inputs):
        """
        Calculate cost
        """
        Wv_b = self.be.dot(self.W.T, inputs) + self.b_hidden
        energy = self.be.empty((1, self.be.bsz))
        energy[:] = -self.be.dot(self.b_visible.T, inputs) - self.be.sum(self.be.log(1 + self.be.exp(Wv_b)), axis=0)
        return energy

    def get_pseudolikelihood(self, inputs):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        if not hasattr(self, 'bit_i_idx'):
            self.bit_i_idx = 0
        else:
            self.bit_i_idx = (self.bit_i_idx + 1) % self.n_visible

        # binarize the input image by rounding to nearest integer
        xi = self.be.array(inputs.get() >= 0.5)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx]
        xi_flip = xi.copy(xi)
        xi_flip[self.bit_i_idx] = 1 - xi_flip[self.bit_i_idx]

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = self.be.empty((1, 1))
        cost[:] = self.be.mean(self.n_visible * self.be.log(self.sigmoid(fe_xi_flip - fe_xi)))
        return cost.get()[0, 0]


class RBMLayerWithLabels(RBMLayer):
    """
    Implementation of an RBM layer with combination of multinomial label variables
    and Bernouli feature variables.

    n_hidden (int): number of hidden Bernouli variables
    n_classes (int): number of classes
    n_duplicates (int): number of multinomial label variables.
        Each class is represented by n identical variables (Bernouli).

    """

    def __init__(self, n_hidden, n_classes, n_duplicates=1, init=None, name="RBMLayerWithLabels",
                 persistant=False, kPCD=1, use_fast_weights=False,
                 sparse_target=0, sparse_damping=0, sparse_cost=0,
                 collect_zero_signal=True):
        super(RBMLayerWithLabels, self).__init__(n_hidden, init=init, name=name, persistant=persistant, kPCD=kPCD,
                                                 use_fast_weights=use_fast_weights, sparse_target=sparse_target,
                                                 sparse_cost=sparse_cost, sparse_damping=sparse_damping,
                                                 collect_zero_signal=collect_zero_signal)
        self.n_classes = n_classes
        self.n_duplicates = n_duplicates

        self.fast_W = None
        self.fast_b_hidden = 0
        self.fast_b_visible = 0

    def allocate(self, shared_outputs=None):
        super(RBMLayerWithLabels, self).allocate(shared_outputs)
        if self.use_fast_weights:
            self.fast_W = self.be.zeros(self.weight_shape)
            self.fast_dW = self.be.zeros(self.weight_shape)
            self.fast_b_hidden = self.be.zeros(self.b_hid_shape)
            self.fast_db_hidden = self.be.zeros(self.b_hid_shape)
            self.fast_b_visible = self.be.zeros(self.b_vis_shape)
            self.fast_db_visible = self.be.zeros(self.b_vis_shape)
        else:
            self.fast_W = 0
            self.fast_b_hidden = 0
            self.fast_b_visible = 0

    def configure(self, in_obj):
        super(RBMLayerWithLabels, self).configure(in_obj)

        self.weight_shape = (self.weight_shape[0] + self.n_classes, self.weight_shape[1])
        self.b_vis_shape = (self.b_vis_shape[0] + self.n_classes, self.b_vis_shape[1])
        return self

    def fprop(self, inputs, inference=False, labels=None, weights=None):
        """
        Calculate hidden units
        """
        if labels is None:
            ohe_labels = np.zeros((self.n_classes, self.be.bsz))
        else:
            ohe_labels = label2binary(labels, self.n_classes)

        v_units = self.be.array(np.vstack((ohe_labels, inputs.get())))

        return super(RBMLayerWithLabels, self).fprop(v_units, inference=inference, weights=weights)

    def bprop(self, hidden_units, alpha=None, beta=None, weights=None):
        """
        CD1 backward pass (negative phase)
        Returns probability of visible units
        """
        visible_proba = super(RBMLayerWithLabels, self).bprop(hidden_units, alpha, beta, weights)
        return visible_proba[self.n_classes:]

    def hidden_probability(self, inputs, labels=None, fast_weights=0, fast_b_hidden=0, weights=None):
        """
        Calculate P(h | v)
        """
        if weights is None:
            weights = self.W

        weights = _calc_optree(weights + fast_weights, self.be)

        v_units = inputs
        if inputs.shape[0] != self.W.shape[0]:
            if not labels is None:
                ohe_labels = label2binary(labels, self.n_classes)
                v_units = self.be.array(np.vstack((ohe_labels, inputs.get())))

        if v_units.shape[0] == self.W.shape[0]:
            weights[:self.n_classes] *= self.n_duplicates
        else:
            weights = weights[self.n_classes:]

        hidden_preacts = self.be.dot(weights.T, v_units) + self.b_hidden + fast_b_hidden
        hidden_proba = self.sigmoid(hidden_preacts)
        return hidden_proba

    def visible_probability(self, hidden_units, fast_weights=0, fast_b_visible=0, weights=None):
        """
        Calculate P(v|h)
        """
        if weights is None:
            weights = self.W
        weights = weights + fast_weights
        visible_preacts = _calc_optree(self.be.dot(weights, hidden_units) + self.b_visible + fast_b_visible,
                                       self.be)
        visible_probability = self.be.empty_like(visible_preacts)
        visible_probability[self.n_classes:] = self.sigmoid(visible_preacts[self.n_classes:])

        # TODO: chekc the axis
        temp_exponential = self.be.exp(visible_preacts[:self.n_classes] -
                                       self.be.max(visible_preacts[:self.n_classes], axis=0))
        visible_probability[:self.n_classes] = temp_exponential / self.be.sum(temp_exponential, axis=0)
        return visible_probability

    def sample_hiddens(self, visible_units, labels=None, fast_weights=0, fast_b_hidden=0):
        """
        Sample hidden units.
        """
        h_probability = self.hidden_probability(visible_units, labels, fast_weights, fast_b_hidden)
        return h_probability > self.be.array(self.be.rng.uniform(size=h_probability.shape))

    def sample_visibles(self, hidden_units, fast_weights=0, fast_b_visible=0):
        """
        Sample visible units
        """
        v_units = self.visible_probability(hidden_units, fast_weights, fast_b_visible)
        v_units[self.n_classes:] = (v_units[self.n_classes:] >
                                    self.be.array(self.be.rng.uniform(size=v_units[self.n_classes:].shape)))

        v_units_tensor = _calc_optree(v_units, self.be)
        # multinomial distribution with n = 1 (number of trials)
        random_numbers = self.be.rng.uniform(size=self.be.bsz)
        probabilities = v_units_tensor[:self.n_classes].get().cumsum(axis=0)
        for i in xrange(self.n_classes):
            if i == 0:
                v_units_tensor.get()[i] = random_numbers < probabilities[i]
            else:
                v_units_tensor.get()[i] = (random_numbers >= probabilities[i - 1]) & (random_numbers < probabilities[i])
        return v_units_tensor

    def update(self, v_pos, labels=None):
        """
        Calculate gradients
        Inputs:
            v_pos (Tensor): input units (typically given input sample X) of size (n_visibles, batch_size)
            labels (Tensor): either OHE labels of shape (n_classes, batch_size) or just labels of shape (1, batch_size).
                In this case it will be converted to OHE labels.
        Returns:
            (update_W, update_b_hidden, update_b_visible) (tuple): gradients of W, b_hidden, b_visible
        """

        if labels is None:
            raise Exception('"labels" must be provided!')

        ohe_labels = label2binary(labels, self.n_classes)
        v_pos = _calc_optree(v_pos, self.be)
        v_pos_with_labels = self.be.array(np.vstack((ohe_labels, v_pos.get())))

        h_pos = self.hidden_probability(v_pos_with_labels)

        # sparsity
        sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos)

        # negative phase
        if self.persistant:
            if self.chain is None:
                self.chain = self.be.zeros(h_pos.shape)
            chain = self.chain
        else:
            chain = h_pos > self.be.array(self.be.rng.uniform(size=h_pos.shape))

        for k in xrange(self.kPCD):
            if self.persistant:
                v_neg = self.sample_visibles(chain, self.fast_W, self.fast_b_visible)
            else:
                v_neg = self.visible_probability(chain, self.fast_W, self.fast_b_visible)

            h_neg = self.hidden_probability(v_neg, fast_weights=self.fast_W, fast_b_hidden=self.fast_b_hidden)

            chain = h_neg > self.be.array(self.be.rng.uniform(size=h_neg.shape))

        if self.persistant:
            self.chain = _calc_optree(chain, self.be)

        update_W = self._grad(v_pos_with_labels, h_pos) - self._grad(v_neg, h_neg)
        update_b_visible = self.be.mean(v_pos_with_labels - v_neg, axis=-1)
        update_b_hidden = self.be.mean(h_pos - h_neg, axis=-1) - sparsegrads_b_hidden

        result = {'W': update_W / float(self.be.bsz),
                  'b_hidden': update_b_hidden,
                  'b_visible': update_b_visible}
        return result

    def update_for_wake_sleep(self, v_pos, labels=None, persistant=False, kPCD=1):
        """
        Calculate gradients during wake-sleep.
        """

        # positive phase
        h_pos = self.hidden_probability(v_pos, labels)
        h_pos_sample = h_pos > self.be.array(self.be.rng.uniform(size=h_pos.shape))

        # negative phase
        if persistant:
            if self.chain is None:
                self.chain = self.be.zeros(h_pos.shape)
            chain = self.chain
        else:
            chain = h_pos_sample

        for k in xrange(kPCD):
            v_neg = self.visible_probability(chain)

            v_neg_sample = v_neg.copy(v_neg)
            v_neg_sample[self.n_classes:] = (v_neg[self.n_classes:] >
                                             self.be.array(self.be.rng.uniform(size=v_neg[self.n_classes:].shape)))

            h_neg = self.hidden_probability(v_neg)

            chain = h_neg > self.be.array(self.be.rng.uniform(size=h_neg.shape))

        if persistant:
            self.chain = _calc_optree(chain, self.be)

        update_W = self._grad(v_pos, h_pos_sample) - self._grad(v_neg_sample, h_neg)
        update_b_visible = self.be.mean(v_pos - v_neg_sample, axis=-1)
        update_b_hidden = self.be.mean(h_pos_sample - h_neg, axis=-1)

        return update_W / float(self.be.bsz), update_b_hidden, update_b_visible, v_neg, v_neg_sample, h_neg


def label2binary(label, n_classes):
    """
    Convert label to binary vector.
    Labels should be from {0, 1, ..., n_classes} set.
    Input:
        label (Tensor): (1, batch_size) Tensor
        n_classes (int): number of classes
    Returns:
        binary (numpy array): (n_classes, batch_size)-shaped Tensor
    """
    if label.shape[0] == n_classes:
        return label.get()

    if label.shape[0] > 1:
        raise Exception('"label" must 1 x N array!')

    binary = np.zeros((n_classes, label.shape[1]), dtype=np.int32)
    binary[:, label.get()] = 1
    return binary


class ConvolutionalRBMLayer(RBMLayer):
    """
    Convolutional RBM layer implementation.
    Works with volumetric data

    Arguments:
        fshape (tuple(int)): four dimensional shape of convolution window (depth, width, height, n_output_maps)
        strides (Optional[Union[int, dict]]): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_d, str_h and str_w applies to d, h and w dimensions distinctly.  Defaults
            to str_d = str_w = str_h = None
        padding (Optional[Union[int, dict]]): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_d, pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_d = pad_w = pad_h = None
        init (Optional[Initializer]): Initializer object to use for
            initializing layer weights
        name (Optional[str]): layer name. Defaults to "ConvolutionLayer"
    """

    def __init__(self, fshape, strides={}, padding={}, init=None, bsum=False, name="ConvolutionalRBMLayer", parallelism="Data",
                 persistant=False, kPCD=1, use_fast_weights=False,
                 sparse_target=0, sparse_damping=0, sparse_cost=0,
                 collect_zero_signal=True):
        super(ConvolutionalRBMLayer, self).__init__(0, init=init, name=name, parallelism=parallelism, persistant=persistant, kPCD=kPCD,
                                                    use_fast_weights=use_fast_weights, sparse_target=sparse_target,
                                                    sparse_cost=sparse_cost, sparse_damping=sparse_damping,
                                                    collect_zero_signal=collect_zero_signal)

        self.nglayer = None
        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'T': 1, 'D': 1, 'bsum': bsum}  # 3D paramaters

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding

        if isinstance(fshape, tuple):
            fkeys = ('R', 'S', 'K') if len(fshape) == 3 else ('T', 'R', 'S', 'K')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        if isinstance(strides, int):
            strides = {'str_d': strides, 'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_d': padding, 'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.convparams.update(d)

        self.b_vis_shape = None
        self.b_hid_shape = None

    def configure(self, in_obj):
        super(ConvolutionalRBMLayer, self).configure(in_obj)

        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.convparams.update(shapedict)
            self.nglayer = self.be.conv_layer(self.be.default_dtype, **self.convparams)
            (K, M, P, Q, N) = self.nglayer.dimO
            self.out_shape = (K, P, Q) if M == 1 else (K, M, P, Q)

            self.n_hidden = self.nglayer.dimO2[0]

            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)
        if self.convparams['bsum']:
            self.batch_sum_shape = (self.nglayer.K, 1)

        self.b_vis_shape = (reduce(mul, self.in_shape), 1)
        self.b_hid_shape = (self.convparams['K'], 1) # K x 1 x 1 x 1 - number of output feature maps

        return self

    def _grad(self, visible_units, hidden_units):
        """
        Calculate positive part of grad_W

        Inputs:
            visible_units (Tensor): visible units
            hidden_units (Tensor): hidden units (or their probabilities)
        """
        result = self.be.empty_like(self.W)
        visible_units_tensor = _calc_optree(visible_units, self.be)
        hidden_units_tensor = _calc_optree(hidden_units, self.be)
        self.be.update_conv(self.nglayer, visible_units_tensor, hidden_units_tensor, result)
        # TODO: check that division by the batch size is needed. It maybe not because of self.batch_sum
        return result / self.be.bsz

    # def _complex_mean(self, input_array, mean_axes):
    #     """
    #     calculate mean(mean(mean(..., axis=-1), axis=2), axis=3), axis=4)
    #     """
    #     #TODO: this function maybe not needed. In new version it seems to work like in numpy
    #     shape = [dim for dim in input_array.shape]
    #     array_to_average = input_array
    #     for axis in mean_axes:
    #         shape[axis] = 1
    #         intermediate_mean = self.be.empty(shape)
    #         intermediate_mean[:] = self.be.sum(array_to_average, axis=axis)
    #         array_to_average = intermediate_mean

    #     return self.be.array(np.squeeze(intermediate_mean.get()).reshape(-1, 1) / self.be.bsz)

    def update(self, v_pos, labels=None):
        """
        Calculate gradients

        Inputs:
            v_pos (Tensor): input units (typically given input sample X) of shape (n_visibles, batch_size)

        Returns:
            (update_W, update_b_hidden, update_b_visible, zeros_ratio) (tuple): gradients of W, b_hidden, b_visible, zeros_ratio
                zeros_ratio (float) is n_hidden_units / (n_hidden_units - n_zero)
        """
        # positive phase
        h_pos = self.hidden_probability(v_pos)

        # negative phase
        if self.persistant:
            if self.chain is None:
                self.chain = self.be.zeros(h_pos.shape)
            chain = self.chain
        else:
            chain = h_pos > self.be.array(self.be.rng.uniform(size=h_pos.shape))

        for k in xrange(self.kPCD):
            if self.persistant:
                v_neg = self.sample_visibles(chain)
            else:
                v_neg = self.visible_probability(chain)

            h_neg = self.hidden_probability(v_neg)
            chain = h_neg > self.be.array(self.be.rng.uniform(size=h_neg.shape))

        if self.persistant:
            self.chain._assign(chain)

        if not self.collect_zero_signal:
            zero_signal_mask = self.hidden_preacts.get() == 0

            h_pos[zero_signal_mask] = sparse_target
            sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos)
            h_pos[zero_signal_mask] = 0
            h_neg[zero_signal_mask] = 0
            zeros_ratio = zero_signal_mask.size / (zero_signal_mask.size - np.sum(zero_signal_mask))
        else:
            sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos)
            zeros_ratio = 1

        update_W = self._grad(v_pos, h_pos) - self._grad(v_neg, h_neg)
        update_b_visible = self.be.mean(v_pos - v_neg, axis=-1)

        #TODO: maybe this should be mean(mean(mean(...))) like in crbm.m?
        update_b_hidden = _calc_optree(self.be.mean(h_pos - h_neg, axis=-1), self.be)
        update_b_hidden = self.be.sum(update_b_hidden.reshape(self.convparams['K'], -1), axis=-1)
        update_b_hidden -= sparsegrads_b_hidden

        result = {'W': update_W / float(self.be.bsz),
                  'b_hidden': update_b_hidden,
                  'b_visible': update_b_visible,
                  'zeros_ratio': zeros_ratio}
        return result

    def hidden_probability(self, inputs, weights=None):
        """
        Calculate P(h | v)
        Inputs:
            inputs (Tensor): visible units of size (n_visible x batch_size)
            weights (Tensor): weights (optional) of size (n_filters * filter_depth * filter_width, filter_height)
        Returns:
            hidden_probability (Tensor): probability of hidden units (n_filters * n_hidden, batch_size)
        """
        #initialization
        if weights is None:
            weights = self.W

        # calculate operation tree for inputs
        inputs_tensor = _calc_optree(inputs, self.be)

        hidden_conv = self.be.empty((self.n_hidden, self.be.bsz))
        self.be.fprop_conv(self.nglayer, inputs_tensor, weights, hidden_conv)

        b_hidden = self.be.ones((self.convparams['K'], self.nglayer.nOut / self.convparams['K']))
        b_hidden[:] = b_hidden * self.b_hidden
        hidden_preacts = hidden_conv + b_hidden.reshape(-1, 1)
        hidden_proba = self.sigmoid(hidden_preacts)

        return hidden_proba

    def visible_probability(self, hidden_units, weights=None):
        """
        Calculate P(v|h)
        """
        if weights is None:
            weights = self.W

        # calculate operation tree for hidden_units
        hidden_units_tensor = _calc_optree(hidden_units, self.be)

        visible_conv = self.be.empty((self.n_visible, self.be.bsz))
        # TODO: maybe we can interchange loop by one convolution? check this.
        self.be.bprop_conv(layer=self.nglayer, F=weights, E=hidden_units_tensor, grad_I=visible_conv, bsum=self.batch_sum)

        visible_preacts = visible_conv + self.b_visible
        visible_proba = self.sigmoid(visible_preacts)
        return visible_proba

    def get_sparse_grads_b_hidden(self, h_probability):
        if self.sparse_cost == 0:
            return self.be.zeros_like(self.b_hidden)

        if not hasattr(self, 'hidmeans'):
            self.hidmeans = self.be.ones(self.nglayer.dimO[:-1]) * self.sparse_target

        h_probability = h_probability.reshape(self.nglayer.dimO)
        hidden_probability_mean = self.be.empty(self.hidmeans.shape + (1,))
        hidden_probability_mean[:] = self.be.mean(h_probability, axis=-1)
        self.hidmeans[:] = self.sparse_damping * self.hidmeans + (1 - self.sparse_damping) * hidden_probability_mean[:, :, :, 0]

        sparsegrads_b_hidden = sparse_cost * np.squeeze(np.mean(np.mean(np.mean(self.hidmeans.get() - self.sparse_target,
                                                                 axis=1), axis=1), axis=1))
        return self.be.array(sparsegrads_b_hidden.reshape(-1, 1))
