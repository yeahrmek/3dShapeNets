from neon.optimizers.optimizer import GradientDescentMomentum, get_param_list


def get_fast_params(layer):
    '''
    returns a flattened list of params
    '''
    plist = (((layer.fast_W, layer.fast_dW),
             (layer.fast_b_hidden, layer.fast_db_hidden),
             (layer.fast_db_visible, layer.fast_db_visible)), layer.fast_states)
    return plist


class GradientDescentMomentumRBM(GradientDescentMomentum):

    """
    Stochastic gradient descent with momentum
    """

    def _update_params(self, param_list, lrate, scale_factor):
        states = param_list[1]
        for i, (param, grad) in enumerate(param_list[0]):
            param.rounding = self.stochastic_round
            if len(states) <= i:
                states.append((self.be.zeros_like(grad)))
            grad = grad / self.be.bsz
            grad = self.clip_gradient_value(grad, self.gradient_clip_value)

            velocity = states[i]
            velocity[:] = velocity * self.momentum_coef \
                - lrate * (scale_factor * grad + self.wdecay[i] * param)
            param[:] = param + velocity

    def _update_params_fast(self, param_list, fast_param_list, lrate, scale_factor):
        self._update_params(param_list, lrate, scale_factor)

        fast_states = fast_param_list[1]
        for i in xrange(len(param_list[0])):
            param, grad = param_list[0][i]
            fast_param, fast_grad = fast_param_list[0][i]
            fast_param.rounding = self.stochastic_round
            if len(fast_states) <= i:
                fast_states.append((self.be.zeros_like(fast_grad)))
            fast_grad = fast_grad / self.be.bsz
            fast_grad = self.clip_gradient_value(fast_grad, self.gradient_clip_value)

            velocity = fast_states[i]
            velocity[:] = (velocity * self.momentum_coef -
                           lrate * (scale_factor * fast_grad + self.wdecay[i] * param))
            fast_param[:] = 19.0 / 20.0 * fast_param + velocity



    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            layer_list (list): a list of Layer objects to optimize.
            epoch (int): the current epoch, needed for the Schedule object.
        """
        param_list = get_param_list(layer_list)

        scale_factor = self.clip_gradient_norm(param_list, self.gradient_clip_norm)
        lrate = self.schedule.get_learning_rate(self.learning_rate, epoch)

        for i, params in enumerate(param_list):
            if layer_list[i].use_fast_weights:
                fast_params = get_fast_params(layer_list[i])
                self._update_params_fast(params, fast_params, lrate, scale_factor)
            else:
                self._update_params(params, lrate, scale_factor)



