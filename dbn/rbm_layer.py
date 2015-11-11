# Implementation of RBM Layer
# It can be either fully connected or convolutional

import logging
from operator import mul


import numpy as np


from neon import NervanaObject
from neon.backends import Autodiff
from neon.layers.layer import Layer, ParameterLayer, Bias, BatchNorm, Activation
from neon.transforms import Logistic


from conv_layer_grad import ConvLayerGrad, fprop_conv_grad


logger = logging.getLogger(__name__)


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
    """

    def __init__(self, n_hidden, init=None, name="RBMLayer"):
        super(RBMLayer, self).__init__(init, name)
        self.b_hidden = None
        self.b_visible = None

        self.sigmoid = Logistic()

        self.chain = None

        # keep around args in __dict__ for get_description.
        self.n_hidden = n_hidden
        self.n_visible = None

    def init_params(self, shape=None, b_vis_shape=None, b_hid_shape=None):
        """
        Allocate layer parameter buffers and initialize them with the
            supplied initializer.

        Arguments:
            shape (int, tuple): shape to allocate for layer paremeter
                buffers.
        """
        if shape:
            self.W = self.be.empty(shape)
            self.dW = self.be.zeros_like(self.W)
            self.init.fill(self.W)

        if not b_hid_shape is None:
            self.b_hidden = self.be.zeros(b_hid_shape)
            self.db_hidden = self.be.zeros_like(self.b_hidden)

        if not b_vis_shape is None:
            self.b_visible = self.be.zeros(b_vis_shape)
            self.db_visible = self.be.zeros_like(self.b_visible)

    def get_params(self):
        """
        Get layer parameters, gradients, and states for optimization
        """
        return ((self.W, self.dW, self.b_hidden, self.db_hidden,
                self.b_visible, self.db_visible), self.states)

    def init_buffers(self, inputs):
        """
        Helper for allocating output and delta buffers (but not initializing
        them)

        Arguments:
            inputs (Tensor): tensor used for frop inputs, used to determine
                shape of buffers being allocated.
        """
        self.inputs = inputs
        if not self.n_visible:
            self.n_visible = self.inputs.shape[0]
            self.visible_preacts = self.be.empty((self.n_visible, self.be.bsz))
            self.hidden_preacts = self.be.empty((self.n_hidden, self.be.bsz))

        if self.weight_shape is None:
            self.weight_shape = (self.n_visible, self.n_hidden)

        # bias for visible units
        if not hasattr(self, 'b_vis_shape') or (hasattr(self, 'b_vis_shape') and self.b_vis_shape is None):
            self.b_vis_shape = (self.n_visible, 1)

        # bias for hidden units
        if not hasattr(self, 'b_hid_shape') or (hasattr(self, 'b_hid_shape') and self.b_hid_shape is None):
            self.b_hid_shape = (self.n_hidden, 1)

    def fprop(self, inputs, labels=None, weights=None):
        """
        forward propagation
        """
        hidden_proba = self.hidden_probability(inputs, weights=weights)
        hidden_units = self.be.array(hidden_proba.get() >
                                     self.be.rng.uniform(size=hidden_proba.shape))
        return hidden_units, hidden_proba

    def bprop(self, hid_units=None, weights=None):
        """
        CD1 backward pass (negative phase)
        """
        visible_proba = self.visible_probability(hid_units, weights=None)
        visible_units = self.be.array(visible_proba.get() >
                                      self.be.rng.uniform(size=visible_proba.shape))
        return visible_units, visible_proba

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

    def update(self, v_pos, labels=None, persistant=False, kPCD=1, use_fast_weights=False,
               sparse_target=0, sparse_damping=0, sparse_cost=0, collect_zero_signal=True):
        """
        Calculate gradients
        Inputs:
            v_pos (Tensor): input units (typically given input sample X) of size (n_visibles, batch_size)
            labels (Tensor): either OHE labels of shape (n_classes, batch_size) or just labels of shape (1, batch_size).
                In this case it will be converted to OHE labels.
            persistant (bool): whether to use persistant CD
            kPCD (int): number of samples generation during negative phase of CD (CD-k)
            use_fast_weights (bool): whether to use fast weights CD algorithm for learning. Not implemented yet!
            sparse_target (float): desired sparsity
            sparse_damping (float): damping of sparsity parameters
            sparse_cost (float): cost of not matching sparsity target
            collect_zero_signal (bool): whether to use units with 0 signal during learning. Not supported.
        Returns:
            (update_W, update_b_hidden, update_b_visible) (tuple): gradients of W, b_hidden, b_visible
        """
        # positive phase
        h_pos = self.hidden_probability(v_pos)

        # negative phase
        if persistant:
            if self.chain is None:
                self.chain = self.be.zeros(h_pos.shape)
        else:
            self.chain = self.be.array(h_pos.get() > self.be.rng.uniform(size=h_pos.shape))

        for k in xrange(kPCD):
            if persistant:
                v_neg = self.sample_visibles(self.chain)
            else:
                v_neg = self.visible_probability(self.chain)

            h_neg = self.hidden_probability(v_neg)
            self.chain = self.be.array(h_neg.get() > self.be.rng.uniform(size=h_neg.shape))


        if not collect_zero_signal:
            zero_signal_mask = self.hidden_preacts.get() == 0

            h_pos[zero_signal_mask] = sparse_target
            sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos, sparse_target,
                                                                  sparse_damping, sparse_cost)
            h_pos[zero_signal_mask] = 0
            h_neg[zero_signal_mask] = 0
        else:
            sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos, sparse_target,
                                                                  sparse_damping, sparse_cost)

        # update_W = self.be.dot(v_pos, h_pos.T) - self.be.dot(v_neg, h_neg.T)
        update_W = self._grad(v_pos, h_pos) - self._grad(v_neg, h_neg)
        update_b_visible = self.be.mean(v_pos - v_neg, axis=-1)
        update_b_hidden = self.be.mean(h_pos - h_neg, axis=-1) - sparsegrads_b_hidden

        result = {'W': update_W / float(self.be.bsz),
                  'b_hidden': update_b_hidden,
                  'b_visible': update_b_visible}
        if not collect_zero_signal:
            result['zero_signal_mask'] = zero_signal_mask
        return result

    def hidden_probability(self, inputs, labels=None, weights=None):
        """
        Calculate P(h | v)
        """
        #initialization
        self.init_buffers(inputs)
        if self.W is None:
            self.init_params(self.weight_shape)

        if self.b_hidden is None:
            self.init_params(b_hid_shape=self.b_hid_shape)

        if weights is None:
            self.hidden_preacts[:] = self.be.dot(self.W.T, inputs)
        else:
            self.hidden_preacts[:] = self.be.dot(weights.T, inputs)

        hidden_proba = self.be.empty_like(self.hidden_preacts)
        hidden_proba[:] = self.sigmoid(self.hidden_preacts + self.b_hidden)

        return hidden_proba

    def visible_probability(self, hidden_units, weights=None):
        """
        Calculate P(v|h)
        """
        if self.b_visible is None:
            self.init_params(b_vis_shape=self.b_vis_shape)

        if weights is None:
            self.visible_preacts[:] = self.be.dot(self.W, hidden_units) + self.b_visible
        else:
            self.visible_preacts[:] = self.be.dot(weights, hidden_units) + self.b_visible

        visible_proba = self.be.empty_like(self.visible_preacts)
        visible_proba[:] = self.sigmoid(self.visible_preacts)
        return visible_proba

    def sample_hiddens(self, visible_units, labels=None):
        """
        Sample hidden units.
        """
        h_probability = self.hidden_probability(visible_units)
        return self.be.array(h_probability.get() > self.be.rng.uniform(size=h_probability.shape))

    def sample_visibles(self, hidden_units):
        """
        Sample visible units
        """
        v_probability = self.visible_probability(hidden_units)
        return self.be.array(v_probability.get() > self.be.rng.uniform(size=v_probability.shape))

    def get_sparse_grads_b_hidden(self, h_proba, sparse_target=0, sparse_damping=0, sparse_cost=0):

        if sparse_cost == 0:
            return self.be.zeros_like(self.b_hidden)

        if not hasattr(self, 'hidmeans'):
            self.hidmeans = sparse_target * self.be.ones((self.n_hidden, 1))

        hidden_probability_mean = self.be.mean(h_proba, axis=-1)
        self.hidmeans[:] = sparse_damping * self.hidmeans + (1 - sparse_damping) * hidden_probability_mean

        sparsegrads_b_hidden = sparse_cost * (self.hidmeans.get() - sparse_target)
        return self.be.array(sparsegrads_b_hidden.reshape(-1, 1))

    def free_energy(self, inputs):
        """
        Calculate cost
        """
        Wv_b = self.be.empty_like(self.hidden_preacts)
        Wv_b[:] = self.be.dot(self.W.T, inputs) + self.b_hidden
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

    def __init__(self, n_hidden, n_classes, n_duplicates=1, init=None, name="RBMLayerWithLabels"):
        super(RBMLayerWithLabels, self).__init__(n_hidden, init=init, name=name)
        self.n_classes = n_classes
        self.n_duplicates = n_duplicates

        self.fast_W = None
        self.fast_b_hidden = 0
        self.fast_b_visible = 0

    def init_buffers(self, inputs, labels=None):
        """
        Helper for allocating output and delta buffers (but not initializing
        them)

        Arguments:
            inputs (Tensor): tensor used for frop inputs, used to determine
                shape of buffers being allocated.
            labels (Tensor): tensor with labels of inputs, one-hot Encoding
        """

        if not self.n_visible:
            self.n_visible = inputs.shape[0]
            if not labels is None:
                self.n_visible += self.n_classes
            self.visible_preacts = self.be.empty((self.n_visible, self.be.bsz))
            self.hidden_preacts = self.be.empty((self.n_hidden, self.be.bsz))

        if self.weight_shape is None:
            self.weight_shape = (self.n_visible, self.n_hidden)

        # bias for visible units
        if not hasattr(self, 'b_vis_shape') or (hasattr(self, 'b_vis_shape') and self.b_vis_shape is None):
            self.b_vis_shape = (self.n_visible, 1)

        # bias for hidden units
        if not hasattr(self, 'b_hid_shape') or (hasattr(self, 'b_hid_shape') and self.b_hid_shape is None):
            self.b_hid_shape = (self.n_hidden, 1)

    def fprop(self, inputs, labels=None):
        """
        Calculate hidden units
        """
        v_units = inputs
        if labels is None:
            ohe_labels = np.zeros((self.n_classes, self.be.bsz))
        else:
            ohe_labels = label2binary(labels, self.n_classes)

        v_units = self.be.array(np.vstack((ohe_labels, inputs.get())))

        return super(RBMLayerWithLabels, self).fprop(v_units)

    def hidden_probability(self, inputs, labels=None, fast_weights=None, fast_b_hidden=0):
        """
        Calculate P(h | v)
        """
        #initialization
        self.init_buffers(inputs, labels)
        if self.W is None:
            self.init_params(self.weight_shape)

        if self.b_hidden is None:
            self.init_params(b_hid_shape=self.b_hid_shape)

        weights = self.W_labels
        if not fast_weights is None:
            weights[:] = weights + fast_weights

        v_units = inputs
        if not labels is None:
            ohe_labels = label2binary(labels, self.n_classes)
            v_units = self.be.array(np.vstack((ohe_labels, inputs.get())))

        self.hidden_preacts[:] = self.be.dot(weights.T, v_units)

        hidden_proba = self.be.empty_like(self.hidden_preacts)
        hidden_proba[:] = self.sigmoid(self.hidden_preacts + self.b_hidden + fast_b_hidden)

        return hidden_proba

    def visible_probability(self, hidden_units, fast_weights=None, fast_b_visible=0):
        """
        Calculate P(v|h)
        """
        if self.b_visible is None:
            self.init_params(b_vis_shape=self.b_vis_shape)

        if fast_weights is None:
            self.visible_preacts[:] = self.be.dot(self.W, hidden_units) + self.b_visible
            visible_probability = self.be.empty(self.visible_preacts.shape)
            visible_probability[:] = self.sigmoid(self.visible_preacts)
            return visible_probability

        self.visible_preacts[:] = self.be.dot(self.W + fast_weights, hidden_units) + self.b_visible + fast_b_visible
        visible_proba = self.be.empty_like(self.visible_preacts)
        visible_proba[self.n_classes:] = self.sigmoid(self.visible_preacts[self.n_classes:])

        temp_exponential = self.be.exp(self.visible_preacts[:self.n_classes] -
                                       self.be.max(self.visible_preacts[:self.n_classes], axis=0))
        visible_proba[:self.n_classes] = temp_exponential / self.be.sum(temp_exponential, axis=0)
        return visible_proba

    def sample_hiddens(self, visible_units, labels=None, fast_weights=None, fast_b_hidden=0):
        """
        Sample hidden units.
        """
        h_probability = self.hidden_probability(visible_units, labels, fast_weights, fast_b_hidden)
        return self.be.array(h_probability.get() > self.be.rng.uniform(size=h_probability.shape))

    def sample_visibles(self, hidden_units, fast_weights=None, fast_b_visible=0):
        """
        Sample visible units
        """
        v_units = self.visible_probability(hidden_units, fast_weights, fast_b_visible)
        v_units[self.n_classes:] = self.be.array(v_units[self.n_classes:].get() >
                                                 self.be.rng.uniform(size=v_units[self.n_classes:].shape))

        # multinomial distribution with n = 1 (number of trials)
        random_numbers = self.be.rng.uniform(size=self.be.bsz)
        probabilities = v_units[:self.n_classes].get().cumsum(axis=0)
        for i in xrange(self.n_classes):
            if i == 0:
                v_units[i] = random_numbers < probabilities[i]
            else:
                v_units[i] = (random_numbers >= probabilities[i - 1]) & (random_numbers < probabilities[i])
        return v_units

    def update(self, v_pos, labels=None, persistant=False, kPCD=1, use_fast_weights=False,
               sparse_target=0, sparse_damping=0, sparse_cost=0, collect_zero_signal=True):
        """
        Calculate gradients
        Inputs:
            v_pos (Tensor): input units (typically given input sample X) of size (n_visibles, batch_size)
            labels (Tensor): either OHE labels of shape (n_classes, batch_size) or just labels of shape (1, batch_size).
                In this case it will be converted to OHE labels.
            persistant (bool): whether to use persistant CD
            kPCD (int): number of samples generation during negative phase of CD (CD-k)
            use_fast_weights (bool): whether to use fast weights CD algorithm for learning.
            sparse_target (float): desired sparsity
            sparse_damping (float): damping of sparsity parameters
            sparse_cost (float): cost of not matching sparsity target
            collect_zero_signal (bool): whether to use units with 0 signal during learning. Not supported.
        Returns:
            (update_W, update_b_hidden, update_b_visible) (tuple): gradients of W, b_hidden, b_visible
        """

        if labels is None:
            raise Exception('"labels" must be provided!')

        ohe_labels = label2binary(labels, self.n_classes)
        v_pos_with_labels = self.be.array(np.vstack((ohe_labels, v_pos.get())))

        # positive phase
        if use_fast_weights:
            if self.fast_W is None:
                self.init_buffers(v_pos_with_labels)
                if self.W is None:
                    self.init_params(shape=self.weight_shape)

                self.fast_W = self.be.zeros(self.weight_shape)
                self.fast_dW = self.be.zeros(self.weight_shape)
                self.fast_b_hidden = self.be.zeros(self.b_hid_shape)
                self.fast_db_hidden = self.be.zeros(self.b_hid_shape)
                self.fast_b_visible = self.be.zeros(self.b_vis_shape)
                self.fast_db_visible = self.be.zeros(self.b_vis_shape)

            self.W_labels = self.W.copy(self.W)
            self.W_labels[:self.n_classes] = self.W_labels[:self.n_classes] * self.n_duplicates
            temp_fast_W = self.fast_W.copy(self.fast_W)
            temp_fast_W[:self.n_classes] = temp_fast_W[:self.n_classes] * self.n_duplicates

        h_pos = self.hidden_probability(v_pos_with_labels)

        # sparsity
        sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos, sparse_target,
                                                              sparse_damping, sparse_cost)

        # negative phase
        if persistant:
            if self.chain is None:
                self.chain = self.be.zeros(h_pos.shape)
        else:
            self.chain = self.be.array(h_pos.get() > self.be.rng.uniform(size=h_pos.shape))

        for k in xrange(kPCD):
            if persistant:
                v_neg = self.sample_visibles(self.chain, self.fast_W, self.fast_b_visible)
            else:
                v_neg = self.visible_probability(self.chain, self.fast_W, self.fast_b_visible)

            if use_fast_weights:
                h_neg = self.hidden_probability(v_neg, fast_weights=temp_fast_W, fast_b_hidden=self.fast_b_hidden)
            else:
                h_neg = self.hidden_probability(v_neg)

            self.chain = self.be.array(h_neg.get() > self.be.rng.uniform(size=h_neg.shape))


        # update_W = self.be.dot(v_pos_with_labels, h_pos.T) - self.be.dot(v_neg, h_neg.T)
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
        self.W_labels = self.W.copy(self.W)
        self.W_labels[:self.n_classes] = self.W_labels[:self.n_classes] * self.n_duplicates
        h_pos = self.hidden_probability(v_pos)
        h_pos_sample = self.be.array(h_pos.get() > self.be.rng.uniform(size=h_pos.shape))

        # negative phase
        if persistant:
            if self.chain is None:
                self.chain = self.be.zeros(h_pos.shape)
        else:
            self.chain = h_pos_sample

        for k in xrange(kPCD):
            v_neg = self.visible_probability(self.chain)

            v_neg_sample = v_neg.copy(v_neg)
            v_neg_sample[self.n_classes:] = self.be.array(v_neg[self.n_classes:].get() >
                                                          self.be.rng.uniform(size=v_neg[self.n_classes:].shape))

            h_neg = self.hidden_probability(v_neg)

            self.chain = self.be.array(h_neg.get() > self.be.rng.uniform(size=h_neg.shape))


        # update_W = self.be.dot(v_pos, h_pos_sample.T) - self.be.dot(v_neg_sample, h_neg.T)
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


class RBMConvolution3D(RBMLayer):
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

    def __init__(self, fshape, strides={}, padding={}, init=None, name="ConvolutionLayer"):
        super(RBMConvolution3D, self).__init__(reduce(mul, fshape), init=init, name=name)
        self.nglayer = None

        self.nglayer_grad_W = None
        self.nglayer_deconv = None

        self.hidden_preacts = None
        self.visible_preacts = None

        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0}

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding

        if isinstance(fshape, (tuple, list)):
            fshape = {'T': fshape[0], 'R': fshape[1], 'S': fshape[2], 'K': fshape[3]}
        if isinstance(strides, int):
            strides = {'str_d': strides, 'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_d': padding, 'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.convparams.update(d)

        self.n_hidden_units = fshape['T'] * fshape['R'] * fshape['S']

    def init_buffers(self, inputs):
        """
        Helper for allocating output and delta buffers (but not initializing
        them)

        Arguments:
            inputs (Tensor): tensor used for frop inputs, used to determine
                shape of buffers being allocated.
        """
        self.inputs = inputs
        if not self.nglayer:
            assert hasattr(self.inputs, 'lshape')
            self.n_visible = reduce(mul, self.inputs.lshape[:4])
            self.convparams['C'] = self.inputs.lshape[0] # n_input_feature_maps
            self.convparams['D'] = self.inputs.lshape[1] # depth of image
            self.convparams['H'] = self.inputs.lshape[2] # height of image
            self.convparams['W'] = self.inputs.lshape[3] # width of image
            self.convparams['N'] = self.be.bsz # n_images in mini-batch
            self.nglayer = self.be.conv_layer(self.be.default_dtype, **self.convparams)
            self.hidden_preacts = self.be.iobuf(self.nglayer.nOut, self.hidden_preacts)
            self.hidden_preacts.lshape = (self.nglayer.K, self.nglayer.M, self.nglayer.P, self.nglayer.Q)
            # self.deltas = self.be.iobuf(self.inputs.shape[0], self.deltas)

        if not self.nglayer_deconv:
            # deconv layer for convolving H and W_flipped (to obtain V in negative phase)
            self.deconvparams = self.convparams.copy()
            del self.deconvparams['D']
            del self.deconvparams['H']
            del self.deconvparams['W']
            self.deconvparams['C'] = 1 # self.convparams['K']
            self.deconvparams['K'] = 1 # self.convparams['C']
            self.deconvparams['M'] = self.nglayer.M
            self.deconvparams['P'] = self.nglayer.P
            self.deconvparams['Q'] = self.nglayer.Q
            self.nglayer_deconv = self.be.deconv_layer(self.be.default_dtype, **self.deconvparams)
            self.visible_preacts = self.be.iobuf(self.nglayer_deconv.nOut, self.visible_preacts)
            self.visible_preacts.lshape = (self.nglayer_deconv.C, self.nglayer_deconv.D,
                                           self.nglayer_deconv.H, self.nglayer_deconv.W)

        # conv layer and params for convolving V and P(H|V)
        if not self.nglayer_grad_W:
            self.convparams_grad_W = self.convparams.copy()
            self.convparams_grad_W['N'] = self.be.bsz
            self.convparams_grad_W['T'] = self.nglayer.M # depth of filter
            self.convparams_grad_W['R'] = self.nglayer.P # height of filter
            self.convparams_grad_W['S'] = self.nglayer.Q # width of filter
            self.nglayer_grad_W = ConvLayerGrad(self.be, self.be.default_dtype, **self.convparams_grad_W)
            # self.visible_preacts = None
            # self.visible_preacts = self.be.iobuf(self.nglayer_grad_W.nOut, self.visible_preacts)

        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * T * R * S, K)

        # bias for visible units
        if not hasattr(self, 'b_vis_shape') or (hasattr(self, 'b_vis_shape') and self.b_vis_shape is None):
            # c in Matlab code
            self.b_vis_shape = (reduce(mul, self.inputs.lshape[1:4]), 1)

        # bias for hidden units
        if not hasattr(self, 'b_hid_shape') or (hasattr(self, 'b_hid_shape') and self.b_hid_shape is None):
            # b in Matlab code
            self.b_hid_shape = (self.convparams['K'], 1) # K x 1 x 1 x 1 - number of output feature maps

    def _grad(self, visible_units, hidden_units):
        """
        Calculate positive part of grad_W

        Inputs:
            visible_units (Tensor): visible units
            hidden_units (Tensor): hidden units (or their probabilities)
        """
        result = self.be.empty(self.nglayer_grad_W.dimO2)
        #TODO: this should be moved to backend. Currently works only with CPU backend
        fprop_conv_grad(self.nglayer_grad_W, visible_units, hidden_units, result)
        return self.be.divide(result, self.be.bsz, out=result)

    def _complex_mean(self, input_array, mean_axes):
        """
        calculate mean(mean(mean(..., axis=-1), axis=2), axis=3), axis=4)
        """
        shape = [dim for dim in input_array.shape]
        array_to_average = input_array
        for axis in mean_axes:
            shape[axis] = 1
            intermediate_mean = self.be.empty(shape)
            intermediate_mean[:] = self.be.sum(array_to_average, axis=axis)
            array_to_average = intermediate_mean

        return self.be.array(np.squeeze(intermediate_mean.get()).reshape(-1, 1) / self.be.bsz)

    def _update_b_hidden(self, h):
        """
        calculate mean(sum(sum(sum(h_pos - h_neg), axis=1), axis=1, axis=1), axix=-1)
        """
        h_mean = self.be.empty((h.shape[0], 1))
        h_mean[:] = self.be.mean(h, axis=-1)
        h_mean = h_mean.reshape(self.convparams['K'], -1)
        return self.be.sum(h_mean, axis=-1)

    def update(self, v_pos, persistant=False, kPCD=1, use_fast_weights=False,
               sparse_target=0, sparse_damping=0, sparse_cost=0, collect_zero_signal=True):
        """
        Calculate gradients
        Inputs:
            v_pos (Tensor): input units (typically given input sample X) of shape (n_visibles, batch_size)
            persistant (bool): whether to use persistant CD
            kPCD (int): number of samples generation during negative phase of CD (CD-k)
            use_fast_weights (bool): whether to use fast weights CD algorithm for learning. Not implemented yet!
            sparse_target (float): desired sparsity
            sparse_damping (float): damping of sparsity parameters
            sparse_cost (float): cost of not matching sparsity target
            collect_zero_signal (bool): whether to use units with 0 signal during learning
        Returns:
            (update_W, update_b_hidden, update_b_visible, zeros_ratio) (tuple): gradients of W, b_hidden, b_visible, zeros_ratio
                zeros_ratio (float) is n_hidden_units / (n_hidden_units - n_zero)
        """
        # positive phase
        h_pos = self.hidden_probability(v_pos)

        # negative phase
        if persistant:
            if self.chain is None:
                self.chain = self.be.zeros(h_pos.shape)
        else:
            self.chain = self.be.array(h_pos.get() > self.be.rng.uniform(size=h_pos.shape))

        for k in xrange(kPCD):
            if persistant:
                v_neg = self.sample_visibles(self.chain)
            else:
                v_neg = self.visible_probability(self.chain)

            h_neg = self.hidden_probability(v_neg)
            self.chain = self.be.array(h_neg.get() > self.be.rng.uniform(size=h_neg.shape))


        if not collect_zero_signal:
            zero_signal_mask = self.hidden_preacts.get() == 0

            h_pos[zero_signal_mask] = sparse_target
            sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos, sparse_target,
                                                                  sparse_damping, sparse_cost)
            h_pos[zero_signal_mask] = 0
            h_neg[zero_signal_mask] = 0
            zeros_ratio = zero_signal_mask.size / (zero_signal_mask.size - np.sum(zero_signal_mask))
        else:
            sparsegrads_b_hidden = self.get_sparse_grads_b_hidden(h_pos, sparse_target,
                                                                  sparse_damping, sparse_cost)
            zeros_ratio = 1

        update_W = self._conv_grad(v_pos, h_pos) - self._conv_inputs_proba(v_neg, h_neg)
        update_b_visible = self.be.mean(v_pos - v_neg, axis=-1)

        #TODO: maybe this should be mean(mean(mean(...))) like in crbm.m?
        update_b_hidden = self._update_b_hidden(h_pos - h_neg) - sparsegrads_b_hidden

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
        self.init_buffers(inputs)
        if self.W is None:
            self.init_params(self.weight_shape)

        if self.b_hidden is None:
            self.init_params(b_hid_shape=self.b_hid_shape)

        if weights is None:
            self.be.fprop_conv(self.nglayer, inputs, self.W, self.hidden_preacts)
        else:
            self.be.fprop_conv(self.nglayer, inputs, weights, self.hidden_preacts)

        b_hidden = self.be.empty((self.convparams['K'], self.nglayer.nOut / self.convparams['K']))
        b_hidden[:] = self.be.ones(b_hidden.shape) * self.b_hidden

        hidden_proba = self.be.empty_like(self.hidden_preacts)
        hidden_proba[:] = self.sigmoid(self.hidden_preacts + b_hidden.reshape(-1, 1))

        return hidden_proba

    def visible_probability(self, hidden_units, weights=None):
        """
        Calculate P(v|h)
        """
        if self.b_visible is None:
            self.init_params(b_vis_shape=self.b_vis_shape)

        if weights is None:
            weights = self.W

        # TODO: maybe we can interchange loop by one convolution? check this.
        tmp_deconv = self.be.zeros(self.visible_preacts.shape)
        hid_units = hidden_units.reshape(self.convparams['K'], -1, self.nglayer.dimO2[1])
        for filter_idx in xrange(self.convparams['K']):
            self.be.bprop_conv(layer=self.nglayer_deconv,
                               F=weights[:, [filter_idx]],
                               E=hid_units[[filter_idx]],
                               grad_I=tmp_deconv)
            self.visible_preacts[:] = self.visible_preacts + tmp_deconv

        self.visible_preacts[:] = self.visible_preacts + self.b_visible

        visible_proba = self.be.empty(self.visible_preacts.shape)
        visible_proba[:] = self.sigmoid(self.visible_preacts)
        return visible_proba

    def get_sparse_grads_b_hidden(self, h_probability, sparse_target=1, sparse_damping=0, sparse_cost=1):

        if not hasattr(self, 'hidmeans'):
            self.hidmeans = self.be.ones(self.nglayer.dimO[:-1])
            self.hidmeans[:] *= sparse_target

        h_probability = h_probability.reshape(self.nglayer.dimO)
        hidden_probability_mean = self.be.empty(self.hidmeans.shape + (1,))
        hidden_probability_mean[:] = self.be.mean(h_probability, axis=-1)
        self.hidmeans[:] = sparse_damping * self.hidmeans + (1 - sparse_damping) * hidden_probability_mean[:, :, :, 0]

        sparsegrads_b_hidden = sparse_cost * np.squeeze(np.mean(np.mean(np.mean(self.hidmeans.get() - sparse_target,
                                                                 axis=1), axis=1), axis=1))
        return self.be.array(sparsegrads_b_hidden.reshape(-1, 1))

    def cost(self, inputs):
        """
        Calculate cost
        """
        hidden_probability




