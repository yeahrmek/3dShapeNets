# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import numpy as np

from neon import NervanaObject
from neon.transforms import CrossEntropyBinary, Logistic
from neon.util.persist import load_obj
from neon.layers import Merge, Activation
from neon.data import DataIterator


class RBM(NervanaObject):
    """
    Restricted Boltzman Machine class.

    Basic model class which stores a list of layers describing the model. Can train the layer
    weights on a dataset, evaluate on a test set and serialize the mode.
    Additional functionality can be added to fit through callback functions.

    Arguments:
        layers (list): List of layers that compose a model.
        name (str): Model name.  Defaults to "model"
        optimizer (Optimizer): Optimizer object which defines the learning rule
                               for updating model parameters (ie DescentMomentum, AdaDelta)
    """

    def __init__(self, layers=[], name="model", optimizer=None):
        super(RBM, self).__init__(name)
        self.optimizer = optimizer
        self.params = None
        self.states = None
        self.epoch_index = 0
        self.ws_epoch_index = 0
        self.finished = False

        self.layers = []
        self.layers_to_optimize = []

        for layer in layers:
            if isinstance(layer, list):
                self.layers.extend(layer)
            else:
                self.layers.append(layer)

        for layer in self.layers:
            if layer.has_params:
                self.layers_to_optimize.append(layer)

            elif isinstance(layer, Merge):
                self.layers_to_optimize += layer.layers_to_optimize

    def set_shortcut(self):
        # infer whether bprop shortcut can be used on final activation
        # self.cost should be set to run this otherwise do nothing
        lastlayer = self.layers[-1]
        try:
            if self.cost.costfunc.__class__ is CrossEntropyBinary:
                if (lastlayer.__class__ is Activation and
                   lastlayer.transform.__class__ is Logistic):
                    lastlayer.transform.set_shortcut(True)
        except:
            # if any attributes are not set or any other exception
            # is thrown leave transform.shortcut as is (do nothing)
            pass

    def load_weights(self, weight_path):
        """
        Loads the layer weights saved in weight_path from serialize().

        Arguments:
            weight_path (str): File containing serialized python dict with layer
                               weights and states.
        """
        pdict = load_obj(weight_path)
        self.epoch_index = pdict['epoch_index']

        param_layers = [l for l in self.layers_to_optimize]
        param_dict_list = pdict['layer_params_states']
        for l, ps in zip(param_layers, param_dict_list):
            l.set_params(ps['params'])
            if 'states' in ps:
                l.set_states(ps['states'])

    def fit(self, dataset, optimizer, num_epochs, callbacks):
        """
        Trains the model parameters on a dataset by minimizing the cost function through
        gradient descent and updates the layer weights according to a learning rule
        defined in optimizer.

        Arguments:
            dataset (iterator): An iterable of minibatches where each
                element is an x where x is the input data
                x is of dimension (feature_size, batch_size)
                Length of the iterator is num_batches which is num_data / batch_size
            cost (Cost): Defines the function which the model is minimizing based
                on the output of the last layer and the input labels
                Default (should be, but actually there is no default value, it should be passed): Energy
            optimizer (Optimizer): Defines the learning rule for updating the model parameters
                Actually, currently it a dict of optimization parameters:
                keys: weight_decay, float
                      momentum, (float, float)
                      learning_rate
                      step_config --- float in [0, 1], configure number of epoch the momentum has to be changed.
            num_epochs: Number of times to iterate over the dataset.
        """

        self.set_shortcut()  # infer if bprop shortcut can be used
        self.optimizer = optimizer
        self.total_cost = self.be.empty((1, 1))
        self.num_epochs = num_epochs

        if not 'persistant' in self.optimizer:
            self.optimizer['persistant'] = False

        if not 'kPCD' in self.optimizer:
            self.optimizer['kPCD'] = 1

        if not 'use_fast_weights' in self.optimizer:
            self.optimizer['use_fast_weights'] = False

        if not 'collect_zero_signal' in self.optimizer:
            self.optimizer['collect_zero_signal'] = True

        self.update_params = {'use_fast_weights': self.optimizer['use_fast_weights'],
                              'persistant': self.optimizer['persistant'],
                              'kPCD': self.optimizer['kPCD'],
                              'sparse_target': self.optimizer['sparse_target'],
                              'sparse_cost': self.optimizer['sparse_cost'],
                              'sparse_damping': self.optimizer['sparse_damping'],
                              'collect_zero_signal': self.optimizer['collect_zero_signal']}

        callbacks.on_train_begin(num_epochs)

        while self.epoch_index < num_epochs and not self.finished:

            callbacks.on_epoch_begin(self.epoch_index)

            self._epoch_fit(dataset, callbacks)

            callbacks.on_epoch_end(self.epoch_index)

            self.epoch_index += 1

        callbacks.on_train_end()

    def _epoch_fit(self, dataset, callbacks):
        """
        Helper function for fit which performs training on a dataset for one epoch.

        Arguments:
            dataset (iterable): Dataset iterator to perform fit on
        """
        epoch = self.epoch_index
        self.total_cost[:] = 0
        print "current implementation trains only one layer. Create model for each layer separately, and then stack them"

        lr = self.optimizer['learning_rate']
        weight_decay = self.optimizer['weight_decay']
        sparse_damping = self.optimizer['sparse_damping']
        sparse_cost = self.optimizer['sparse_cost']
        sparse_target = self.optimizer['sparse_target']

        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):

            callbacks.on_minibatch_begin(epoch, mb_idx)

            # TODO: implement schedule
            momentum = self.optimizer['momentum'][-1]
            if self.epoch_index < self.num_epochs * self.optimizer['step_config']:
                momentum = self.optimizer['momentum'][0]

            for l in self.layers_to_optimize:
                # this part for sparsity
                update = l.update(x, **self.update_params)

                l.dW[:] = momentum * l.dW + lr * (update['W'] - weight_decay * l.W)
                # TODO: check this. Maybe division by n_visible is not correct.
                l.db_hidden[:] = momentum * l.db_hidden + lr * update['b_hidden']
                l.db_visible[:] = momentum * l.db_visible + lr * update['b_visible'] # / l.n_visible

                l.W[:] = l.W + l.dW
                l.b_visible[:] = l.b_visible + l.db_visible
                l.b_hidden[:] = l.b_hidden + l.db_hidden

                #update fast weights
                if self.optimizer['use_fast_weights']:
                    l.fast_dW[:] = momentum * l.fast_dW + lr * (update['W'] - weight_decay * l.W)
                    l.fast_db_visible = momentum * l.fast_db_visible + lr * update['b_visible']
                    l.fast_db_hidden = momentum * l.fast_db_hidden + lr * update['b_hidden']

                    l.fast_W[:] = 19.0 / 20 * l.fast_W + l.fast_dW
                    l.fast_db_visible = 19.0 / 20 * l.fast_b_visible + l.fast_db_visible
                    l.fast_db_hidden = 19.0 / 20 * l.fast_b_hidden + l.fast_db_hidden

            callbacks.on_minibatch_end(epoch, mb_idx)

    def fprop(self, x, labels=None):
        """
        Forward propagates a minibatch x through the model.

        Arguments:
            x (Tensor): Input minibatch data
            pos_stat (bool): Flag for collecting positiva statistics.
                If set to True returns hidden probability of the last layer
        Returns:
            Tensor: the output of the final layer in the model
        """
        for l in self.layers:
            x, proba = l.fprop(x, labels)
        return x, proba

    def bprop(self, hidden_units, do_acts=True, neg_stat=False, neg_stat_params=None):
        """
        Back propagates the error of a minibatch through the model.

        Arguments:
            hidden_units (Tensor): Hidden units
            do_acts (bool): Whether to compute the output deltas of layer. The first layer
                does not need to compute output deltas and so do_acts is set to False.
            neg_stat (bool): Whether to collect negative statistics.
                If set to True returns visible probability of the last layer
        """
        if neg_stat:
            if neg_stat_params is None:
                neg_stat_params = {}
            return self.layers[-1].neg_stat(**neg_stat_params)

        x_reconstructed = hidden_units
        for l in reversed(self.layers):
            x_reconstructed = l.bprop(x_reconstructed)
        return x_reconstructed

    def eval(self, dataset, metric):
        """
        Evaluates a model on a dataset according to an input metric.

        Arguments:
            datasets (iterable): dataset to evaluate on.
            metric (Cost): what function to evaluate dataset on.
        """
        running_error = 0.0
        nprocessed = 0
        dataset.reset()
        for x, t in dataset:
            x = self.fprop(x)

            # This logic is for handling partial batch sizes at the end of the dataset
            bsz = min(dataset.ndata - nprocessed, self.be.bsz)
            metric(x, t)
            running_error += metric.outputs.get()[:, :bsz].sum()
            nprocessed += bsz
        running_error /= nprocessed
        return running_error

    def predict(self, x):
        """
        Make prediction
        """

        x = self.be.array(x)
        prediction = []
        for i in xrange(x.shape[0]):
            prediction += [self.fprop(x[i]).get()]

        return np.array(prediction)

    def gibbs_sampling(self, inputs):
        """
        Perform Gibbs sampling
        """
        hidden, _ = self.fprop(inputs)
        visible, _ = self.bprop(hidden)
        return visible

    def get_description(self):
        """
        Gets a description of the model required to reconstruct the model with
        no weights like from a yaml file.

        Returns:
            dict: Description of each component of the model.
        """
        pdict = dict()
        pdict['backend'] = 'gpu'
        pdict['cost'] = self.cost.costfunc.__class__.__name__
        pdict['layers'] = [l.get_description() for l in self.layers]
        if self.optimizer:
            pdict['optimizer'] = self.optimizer.get_description()
        return pdict

    # serialize tells how to write out the parameters we've learned so
    # far and associate them with layers. it can ignore layers with no
    # learned parameters. the model stores states to pass to the
    # optimizers.  if we're saving the model out for inference, we
    # don't need to remember states.

    def serialize(self, keep_states=True):
        """
        Creates a dictionary storing the layer parameters and epochs complete.

        Arguments:
            keep_states (bool): Whether to save optimizer states.

        Returns:
            dict: Model data including layer parameters and epochs complete.
        """

        pdict = dict()
        params_states = [l.get_params_serialize(keep_states) for l in self.layers_to_optimize]
        pdict['layer_params_states'] = params_states
        # start training again on the next epoch
        pdict['epoch_index'] = self.epoch_index + 1
        return pdict

    def wake_sleep(self, dataset, optimizer, num_epochs, callbacks):
        """
        Fine tune RBM network using wake-sleep algorithm
        """

        self.ws_optimizer = optimizer

        # reset persistant chain of the top rbm layer
        if not layers_to_optimize[-1].chain is None:
            layers_to_optimize[-1].chain = None

        callbacks.on_train_begin(num_epochs)

        self._init_wake_sleep()

        while self.ws_epoch_index < num_epochs and not self.finished:

            callbacks.on_epoch_begin(self.epoch_index)

            self._epoch_wake_sleep(dataset, callbacks)

            callbacks.on_epoch_end(self.epoch_index)

            self.ws_epoch_index += 1

        callbacks.on_train_end()

    def _epoch_wake_sleep(self, dataset, callbacks):
        """
        """
        epoch = self.ws_epoch_index
        self.total_cost[:] = 0

        #TODO: walkaround for optimization params: do checks, defauts, etc.

        self.ws_update_params = {'use_fast_weights': self.ws_optimizer['use_fast_weights'],
                                 'persistant': self.optimizer['persistant'],
                                 'kPCD': self.ws_optimizer['kPCD'],
                                 'sparse_target': self.ws_optimizer['sparse_target'],
                                 'sparse_cost': self.ws_optimizer['spars2e_cost'],
                                 'sparse_damping': self.ws_optimizer['sparse_damping']}

        momentum = self.ws_optimizer['momentum'][0]
        lr = self.ws_optimizer['learning_rate']
        weight_decay = self.ws_optimizer['weight_decay']
        sparse_damping = self.ws_optimizer['sparse_damping']
        sparse_cost = self.ws_optimizer['sparse_cost']
        sparse_target = self.ws_optimizer['sparse_target']

        reconstruction_errors = []
        Pseudolikelihood = []
        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):

            callbacks.on_minibatch_begin(epoch, mb_idx)

             # TODO: implement schedule
            momentum = self.ws_optimizer['momentum'][-1]
            if self.epoch_index < self.num_epochs * self.ws_optimizer['step_config']:
                momentum = momentum[0]

            # wake phase. [BOTTOM-UP PASS] Assuming recognition weight is correct, and update the generative weight.
            n_layers = len(self.layers_to_optimize)

            wake_states = [] * n_layers
            wake_states[0] = x
            for i, l in enumrate(self.layers_to_optimize[:-1]):
                wake_states[i + 1], _ = l.fprop(wake_states[i], labels=t, weights=l.UW)

            # updates for top rbm. It slightly differs from
            update_top_W, update_top_b_visible, update_top_b_hidden, v_neg_top, v_neg_sample_top, h_neg_top = \
                layers_to_optimize[-1].update_for_wake_sleep(x, t, **self.update_params)


            # sleep phase. [TOP-DOWN PASS] Assuming generative weight is correct, and update the recognition weight.
            sleep_activations = [] * (n_layers + 1)
            sleep_activations[-1] = h_neg_top
            sleep_activations[-2] = v_neg_top[layers_to_optimize[-1].n_classes:]
            sleep_states = [] * n_layers
            sleep_statesp[-1] = layers_to_optimize[-1].chain
            sleep_states[-2] = v_neg_sample_top[layers_to_optimize[-1].n_classes:]

            # generating top - down
            for i, l in reversed(list(enumerate(layers_to_optimize[:-1]))):
                sleep_states[i], sleep_activations[i] = l.bprop(sleep_states[i + 1], weights=l.DW)
            sleep_states[0] = sleep_activations[0]

            # prediction
            p_gen = [] * (n_layers - 1)
            for i, l in enumerate(layers_to_optimize[:-1]):
                _, p_gen[i] = l.bprop(wake_states[i + 1], weights=l.DW)

            p_rec = [] * (n_layers - 1)
            for i, l in enumerate(layers_to_optimize[:-1]):
                _, p_rec[i] = l.fprop(sleep_states[i])


            ## update parameters
            # top rbm
            layers_to_optimize[-1].dW[:] = momentum * layers_to_optimize[-1].dW + lr  * update_top_W
            layers_to_optimize[-1].db_hidden[:] = momentum * layers_to_optimize[-1].db_hidden + lr  * update_b_hidden
            layers_to_optimize[-1].db_visible[:] = momentum * layers_to_optimize[-1].db_visible + lr * update_b_visible

            layers_to_optimize[-1].W[:] = layers_to_optimize[-1].W[:] + layers_to_optimize[-1].dW
            layers_to_optimize[-1].b_hidden[:] = layers_to_optimize[-1].b_hidden + layers_to_optimize[-1].db_hidden
            layers_to_optimize[-1].b_visible[:] = layers_to_optimize[-1].b_visible + layers_to_optimize[-1].db_visible


            #TODO: it looks like convolution in Matlab kConv_weights divides
            # the result by the batch size and by the size of filter. Check this and make corresponding fix
            # in the code below. Also, don't forget to review code of rbm_layer.py

            #generative weights
            for i, l in enumerate(layers_to_optimize[:-1]):
                l.dDW[:] = momentum * l.dDW + lr * l.n_hidden_units * \
                           l._grad(wake_states[i] - p_gen[i], wake_states[i + 1]) / l.be.bsz
                l.db_visible[:] = momentum * l.db_visible + lr * self.be.mean(wake_states[i] - p_gen[i], axis=-1)
                l.dW[:] = l.dW + l.dDW
                l.b_visible[:] = l.b_visible + l.db_visible

            #recognition weights
            for i, l in enumerate(layers_to_optimize[:-1]):
                l.dUW[:] = momentum * l.dUW + lr * l.n_hidden_units * \
                           l._grad(sleep_activations[i], sleep_states[i + 1] - p_rec[i]) / l.be.bsz
                l.db_visible[:] = momentum * l.db_visible + lr * self.be.mean(wake_states[i] - p_gen[i], axis=-1)
                l.dW[:] = l.dW + l.dDW
                l.b_visible[:] = l.b_visible + l.db_visible

            callbacks.on_minibatch_end(epoch, mb_idx)

    def _init_wake_sleep(self):
        """
        Initialize layers for fine-tuning (wake-sleep algorithm)
        """
        for l in self.layers_to_optimize[:-1]:
            l.UW = l.W.copy(l.W)
            l.DW = l.W.copy(l.W)

            l.db_hidden = l.be.zeros_like(l.db_hidden)
            l.db_visible = l.be.zeros_like(l.db_visible)
            l.dUW = l.be.zeros_like(l.uW)
            l.dDW = l.be.zeros_like(l.dW)

        self.layers_to_optimize[-1].db_hidden = l.be.zeros_like(self.layers_to_optimize[-1].db_hidden)
        self.layers_to_optimize[-1].db_visible = l.be.zeros_like(self.layers_to_optimize[-1].db_visible)
        self.layers_to_optimize[-1].dW = l.be.zeros_like(self.layers_to_optimize[-1].dW)
