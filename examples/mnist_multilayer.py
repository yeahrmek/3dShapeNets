import sys
sys.path.append('../dbn/')


from rbm_layer import ConvolutionalRBMLayer, RBMLayerWithLabels, RBMLayer


from neon.backends import gen_backend
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost
from neon.transforms import Softmax, CrossEntropyMulti, Misclassification, Logistic, CrossEntropyBinary, SumSquared
from neon.data import DataIterator
from neon.data.datasets import fetch_dataset
from rbm import RBM
from neon.models import Model
from neon.data.datasets import dataset_meta
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser


import numpy as np
from matplotlib import pyplot
import os
import gzip
import cPickle


def load_mnist(path=".", normalize=True):
    """
    Fetch the MNIST dataset and load it into memory.

    Args:
        path (str, optional): Local directory in which to cache the raw
                              dataset.  Defaults to current directory.
        normalize (bool, optional): whether to scale values between 0 and 1.
                                    Defaults to True.

    Returns:
        tuple: Both training and test sets are returned.
    """
    mnist = dataset_meta['mnist']

    filepath = os.path.join(path, mnist['file'])
    if not os.path.exists(filepath):
        fetch_dataset(mnist['url'], mnist['file'], filepath, mnist['size'])

    with gzip.open(filepath, 'rb') as mnist:
        (X_train, y_train), (X_test, y_test) = cPickle.load(mnist)
        # X_train = X_train.reshape(-1, 1, 28, 28)
        # X_test = X_test.reshape(-1, 1, 28, 28)
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)

        # X_train = X_train[:, :100]
        # X_test = X_test[:, :100]

        if normalize:
            X_train = X_train / 255.
            X_test = X_test / 255.

        return (X_train, y_train), (X_test, y_test), 10


# parse the command line arguments
parser = NeonArgparser(__doc__)

args = parser.parse_args()
args.backend='cpu'
# args.batch_size = 128
args.rng_seed = 0
args.device_id = 0
args.datatype = np.float32

# hyperparameters
batch_size = args.batch_size
num_epochs = args.epochs
n_hidden = 100

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id)

(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)

# setup a training set iterator
train_set = DataIterator(X_train, y=y_train, nclass=nclass, lshape=(1, 28, 28))
# setup a validation data set iterator
valid_set = DataIterator(X_test, y=y_test, nclass=nclass, lshape=(1, 28, 28))

# setup weight initialization function
init_norm = GlorotUniform() # Gaussian(loc=0.0, scale=0.01)

# setiup model layers
n_filters = 2
fshape = 5
layers = [# ConvolutionalRBMLayer(fshape=(1, fshape, fshape, n_filters),
          #                       init=init_norm, strides={'str_d': 1, 'str_h': 2, 'str_w': 2},
          #                       sparse_cost=0.0, sparse_damping=0.0, sparse_target=0.0),
          RBMLayer(n_hidden=100, init=init_norm,
                   sparse_cost=0.0, sparse_damping=0.0, sparse_target=0.0),
          RBMLayerWithLabels(n_hidden=50, init=init_norm, n_classes=nclass, use_fast_weights=True)]


# setup optimizer
optimizer = {'momentum': [0],
             'step_config': 1,
             'learning_rate': 0.1,
             'weight_decay': 0}

# initialize model object
rbm = RBM(layers=layers)

if args.model_file:
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    logger.info('loading initial model state from %s' % args.model_file)
    rbm.load_weights(args.model_file)

# setup standard fit callbacks
callbacks = Callbacks(rbm, train_set, output_file=args.output_file,
                      progress_bar=args.progress_bar)

# add a callback ot calculate

if args.serialize > 0:
    # add callback for saving checkpoint file
    # every args.serialize epchs
    checkpoint_schedule = args.serialize
    checkpoint_model_path = args.save_path
    callbacks.add_serialize_callback(checkpoint_schedule, checkpoint_model_path)

rbm.fit(train_set, optimizer=optimizer, num_epochs=num_epochs, callbacks=callbacks)

for mb_idx, (x_val, y_val) in enumerate(valid_set):
    hidden = rbm.fprop(x_val)
    break
