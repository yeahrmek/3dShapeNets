import sys
sys.path.append('../dbn')


from neon.backends import gen_backend
from dataset import ModelNetDataset
from rbm_layer import ConvolutionalRBMLayer, RBMLayer, RBMLayerWithLabels
from rbm import RBM
from neon.callbacks.callbacks import Callbacks
from neon.initializers import GlorotUniform


import numpy as np
import time


# path = '../../../../../Work/3DShapeNets/volumetric_data/'
# path = '../../3DShapeNetsMatlab/volumetric_data'
path = '../../3DShapeNets/volumetric_data'
classes = ['bathtub', 'bed', 'chair', 'desk']
n_classes = len(classes)

be = gen_backend(backend='cpu',
                 batch_size=32,
                 rng_seed=0,
                 device_id=0)

data = ModelNetDataset(path, classes=classes, data_size=30, lshape=(1, 30, 30, 30))

# train first layer
optimizers = [GradientDescentMomentum(learning_rate=0.01, momentum_coef=[0.5, 0.9], wdecay=1e-5)]
parameters = {'momentum': [0,5, 0.9],
              'step_config': 0.25,
              'learning_rate': 0.01,
              'weight_decay': 1e-5,
              'sparse_damping': 0,
              'sparse_cost': 0.001,
              'sparse_target': 0.01,
              'persistant': False,
              'kPCD': 1,
              'use_fast_weights': False
              }
n_epochs = 1

init = GlorotUniform()

# it seems that the data have shape 30x30x30, though I think it should be 24 with padding=2
layers = [RBMConvolution3D([6, 6, 6, 48], strides=2, padding=0, init=init, name='l1_conv'),
          RBMConvolution3D([5, 5, 5, 160], strides=2, padding=0, init=init, name='l2_conv'),
          RBMConvolution3D([4, 4, 4, 512], strides=2, padding=0, init=init, name='l3_conv'),
          RBMLayer(1200, init=init, name='l4_rbm'),
          RBMLayerWithLabels(4000, n_classes, name='l4_rbm_with_labels')]




rbm = RBM(layers=layers)

# callbacks = Callbacks(rbm, data, output_file='./output.hdf5')
 callbacks = Callbacks(rbm, data)


t = time.time()
rbm.fit(data, optimizer=parameters, num_epochs=n_epochs, callbacks=callbacks)
t = time.time() - t
print "Training time: ", t
