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


from neon.transforms import CrossEntropyBinary, Logistic
from neon.util.persist import load_obj
from neon.data import DataIterator
from neon.models.model import Model


class RBM(Model):
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

        if isinstance(num_epochs, int):
            num_epochs = [num_epochs] * len(self.layers.layers_to_optimize)
        self.num_epochs = num_epochs

        self.optimizer = optimizer

        from neon.layers import GeneralizedCost
        from neon.transforms.cost import SumSquared
        self.cost = GeneralizedCost(costfunc=SumSquared())

        self.initialize(dataset, self.cost)
        self.total_cost = self.be.empty((1, 1))

        self.layer_being_trained = 0

        callbacks.on_train_begin(num_epochs[0])

        for i in xrange(len(self.layers.layers_to_optimize)):

            while self.epoch_index < num_epochs[i] and not self.finished:

                callbacks.on_epoch_begin(self.epoch_index)

                self._epoch_fit(dataset, callbacks)

                callbacks.on_epoch_end(self.epoch_index)

                self.epoch_index += 1

            self.epoch_index = 0
            self.layer_being_trained += 1

        callbacks.on_train_end()

    def _epoch_fit(self, dataset, callbacks):
        """
        Helper function for fit which performs training on a dataset for one epoch.

        Arguments:
            dataset (iterable): Dataset iterator to perform fit on
        """
        epoch = self.epoch_index

        #TODO: implement using Optimizer
        lr = self.optimizer['learning_rate']
        weight_decay = self.optimizer['weight_decay']

        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):

            callbacks.on_minibatch_begin(epoch, mb_idx)

            x = self.fprop(x, labels=t, fprop_to_layer=self.layer_being_trained)

            # TODO: implement schedule
            momentum = self.optimizer['momentum'][-1]
            if self.epoch_index < self.num_epochs * self.optimizer['step_config']:
                momentum = self.optimizer['momentum'][0]

            layer = self.layers.layers_to_optimize[self.layer_being_trained]

            # this part for sparsity
            update = layer.update(x, labels=t)

            layer.dW[:] = momentum * layer.dW + lr * (update['W'] - weight_decay * layer.W)
            # TODO: check this. Maybe division by n_visible is not correct.
            layer.db_hidden[:] = momentum * layer.db_hidden + lr * update['b_hidden']
            layer.db_visible[:] = momentum * layer.db_visible + lr * update['b_visible'] # / layer.n_visible

            layer.W[:] = layer.W + layer.dW
            layer.b_visible[:] = layer.b_visible + layer.db_visible
            layer.b_hidden[:] = layer.b_hidden + layer.db_hidden

            #update fast weights
            if layer.use_fast_weights:
                layer.fast_dW[:] = momentum * layer.fast_dW + lr * (update['W'] - weight_decay * layer.W)
                layer.fast_db_visible[:] = momentum * layer.fast_db_visible + lr * update['b_visible']
                layer.fast_db_hidden[:] = momentum * layer.fast_db_hidden + lr * update['b_hidden']

                layer.fast_W[:] = 19.0 / 20 * layer.fast_W + layer.fast_dW
                layer.fast_db_visible[:] = 19.0 / 20 * layer.fast_b_visible + layer.fast_db_visible
                layer.fast_db_hidden[:] = 19.0 / 20 * layer.fast_b_hidden + layer.fast_db_hidden

            # import pdb
            # pdb.set_trace()
            # from neon.optimizers.optimizer import GradientDescentMomentum
            # opt = GradientDescentMomentum(learning_rate=lr, wdecay=weight_decay, momentum_coef=momentum)
            # opt.optimize(self.layers_to_optimize, epoch=epoch)

            callbacks.on_minibatch_end(epoch, mb_idx)

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

    def fprop(self, x, inference=False, labels=None, fprop_to_layer=None, use_recognition_weights=False):
        """
        Forward propagates a minibatch x through the model.

        Arguments:
            x (Tensor): Input minibatch data
            inference (bool): Flag for performing training or inference
                Only affects batch norm and dropout layers.

        Returns:
            Tensor: the output of the final layer in the model
        """
        hidden_units = x
        if fprop_to_layer is None:
            fprop_to_layer = len(self.layers.layers)

        for l in self.layers.layers[:fprop_to_layer]:
            if use_recognition_weights:
                hidden_units = l.fprop(hidden_units, inference=inference, labels=labels, weights=l.UW)
            else:
                hidden_units = l.fprop(hidden_units, inference=inference, labels=labels)

        return hidden_units

    def gibbs_sampling(self, inputs):
        """
        Perform Gibbs sampling
        """
        hidden = self.fprop(inputs)
        hidden = self.be.array(hidden.get() > self.be.rng.uniform(size=hidden.shape))
        #TOOD: make correct bprop
        visible_optree = self.layers.layers[0].sample_visibles(hidden)
        visible = self.be.empty(visible_optree.shape)
        visible._assign(visible_optree)
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
        self.ws_num_epochs = num_epochs

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

        layers = self.layers.layers_to_optimize
        n_layers = len(layers)

        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):

            callbacks.on_minibatch_begin(epoch, mb_idx)

             # TODO: implement schedule
            momentum = self.ws_optimizer['momentum'][-1]
            if self.ws_epoch_index < self.ws_num_epochs * self.ws_optimizer['step_config']:
                momentum = momentum[0]

            # wake phase. [BOTTOM-UP PASS] Assuming recognition weight is correct, and update the generative weight.
            wake_states = [] * n_layers
            wake_states[0] = x
            for i, l in enumrate(layers[:-1]):
                wake_states[i + 1], _ = l.fprop(wake_states[i], labels=t, weights=l.UW)

            # updates for top rbm. It slightly differs from
            update_top_W, update_top_b_visible, update_top_b_hidden, v_neg_top, v_neg_sample_top, h_neg_top = \
                layers[-1].update_for_wake_sleep(x, t, **self.update_params)


            # sleep phase. [TOP-DOWN PASS] Assuming generative weight is correct, and update the recognition weight.
            sleep_activations = [] * (n_layers + 1)
            sleep_activations[-1] = h_neg_top
            sleep_activations[-2] = v_neg_top[layers[-1].n_classes:]
            sleep_states = [] * n_layers
            sleep_statesp[-1] = layers_to_optimize[-1].chain
            sleep_states[-2] = v_neg_sample_top[layers[-1].n_classes:]

            # generating top - down
            for i, l in reversed(list(enumerate(layers[:-1]))):
                sleep_states[i], sleep_activations[i] = l.bprop(sleep_states[i + 1], weights=l.DW)
            sleep_states[0] = sleep_activations[0]

            # prediction
            p_gen = [] * (n_layers - 1)
            for i, l in enumerate(layers[:-1]):
                _, p_gen[i] = l.bprop(wake_states[i + 1], weights=l.DW)

            p_rec = [] * (n_layers - 1)
            for i, l in enumerate(layers[:-1]):
                _, p_rec[i] = l.fprop(sleep_states[i])


            ## update parameters
            # top rbm
            layers[-1].dW[:] = momentum * layers[-1].dW + lr  * update_top_W
            layers[-1].db_hidden[:] = momentum * layers[-1].db_hidden + lr  * update_b_hidden
            layers[-1].db_visible[:] = momentum * layers[-1].db_visible + lr * update_b_visible

            layers[-1].W[:] = layers[-1].W + layers[-1].dW
            layers[-1].b_hidden[:] = layers[-1].b_hidden + layers[-1].db_hidden
            layers[-1].b_visible[:] = layers[-1].b_visible + layers[-1].db_visible

            #TODO: it looks like convolution in Matlab kConv_weights divides
            # the result by the batch size and by the size of filter. Check this and make corresponding fix
            # in the code below. Also, don't forget to review code of rbm_layer.py

            #generative weights
            for i, l in enumerate(layers[:-1]):
                l.dDW[:] = momentum * l.dDW + lr * l.n_hidden_units * \
                           l._grad(wake_states[i] - p_gen[i], wake_states[i + 1]) / l.be.bsz
                l.db_visible[:] = momentum * l.db_visible + lr * self.be.mean(wake_states[i] - p_gen[i], axis=-1)
                l.DW[:] = l.DW + l.dDW
                l.b_visible[:] = l.b_visible + l.db_visible

            #recognition weights
            #TODO: check this
            for i, l in enumerate(layers[:-1]):
                l.dUW[:] = momentum * l.dUW + lr * l.n_hidden_units * \
                           l._grad(sleep_activations[i], sleep_states[i + 1] - p_rec[i]) / l.be.bsz
                l.db_hidden[:] = momentum * l.db_hidden + lr * self.be.mean(sleep_states[i] - p_rec[i], axis=-1)
                l.UW[:] = l.UW + l.dUW
                l.b_hidden[:] = l.b_hidden + l.db_hidden

            callbacks.on_minibatch_end(epoch, mb_idx)

    def _init_wake_sleep(self):
        """
        Initialize layers for fine-tuning (wake-sleep algorithm)
        """
        for l in self.layers.layers_to_optimize[:-1]:
            l.UW = l.W.copy(l.W)
            l.DW = l.W.copy(l.W)

            l.db_hidden = l.be.zeros_like(l.db_hidden)
            l.db_visible = l.be.zeros_like(l.db_visible)
            l.dUW = l.be.zeros_like(l.uW)
            l.dDW = l.be.zeros_like(l.dW)

        self.layers.layers_to_optimize[-1].db_hidden = l.be.zeros_like(self.layers.layers_to_optimize[-1].db_hidden)
        self.layers.layers_to_optimize[-1].db_visible = l.be.zeros_like(self.layers.layers_to_optimize[-1].db_visible)
        self.layers.layers_to_optimize[-1].dW = l.be.zeros_like(self.layers.layers_to_optimize[-1].dW)
