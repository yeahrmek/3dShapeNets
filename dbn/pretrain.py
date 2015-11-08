import sys
sys.path.append('../dbn')


from neon.backends import gen_backend
from dataset import ModelNetDataset
from rbm_layer import RBMConvolution3D
from rbm import RBM
from neon.callbacks.callbacks import Callbacks
from neon.initializers import GlorotUniform


import numpy as np
import time


# path = '../../../../../Work/3DShapeNets/volumetric_data/'
path = '../../3DShapeNetsMatlab/volumetric_data'
classes = ['bathtub', 'bed', 'chair', 'desk']

be = gen_backend(backend='cpu',
                 batch_size=256,
                 rng_seed=0,
                 device_id=0,
                 default_dtype=np.float32)

data = ModelNetDataset(path, classes=classes, data_size=30, lshape=(1, 30, 30, 30))

# train first layer
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

layers = []

# it seems that the data have shape 30x30x30, though I think it should be 24 with padding=2
layers += [RBMConvolution3D([6, 6, 6, 1], strides=2, padding=0, init=init)]


rbm = RBM(layers=layers)

callbacks = Callbacks(rbm, data, output_file='./output.hdf5')


t = time.time()
rbm.fit(data, optimizer=parameters, num_epochs=n_epochs, callbacks=callbacks)
t = time.time() - t
print "Training time: ", t