from neon import NervanaObject


import pandas as pd
import numpy as np
from scipy.io import loadmat


import os


class ModelNetDataset(NervanaObject):

    """
    This generic class defines an interface to iterate over minibatches of
    data that has been preloaded into memory. This may be used when the
    entire dataset is small enough to fit within memory.
    """

    def __init__(self, path, classes=None, data_size=None, lshape=None, is_train=True, make_onehot=True):
        """
        Implements loading of given data into backend tensor objects. If the
        backend is specific to an accelarator device, the data is copied over
        to that device.

        Args:
            path (str): path to the directory with the dataset
            classes (list): list of strings of names of classes.
            data_size (int): number of voxels in one dimension (it is assumed that the number of voxels is equal for all dimensions)
            is_train (bool): whether the dataset is train or test
            make_onehot (bool, optional): True if y is a label that has to be converted to one hot
                            False if y doesn't need to be converted to one hot
                            (e.g. in a CAE)

        """
        if classes is None:
            raise Exception('"classes" must be provided!')
        if data_size is None:
            raise Exception('"data_size" must be provided')

        self.classes = {}.fromkeys(np.unique(classes).tolist())
        self.n_classes = len(self.classes)
        self.inv_classes = {}.fromkeys(range(self.n_classes))

        self.ndata = 0
        for i, c in enumerate(self.classes):
            self.classes[c] = {}

            self.classes[c]['path'] = os.path.join(path, c, str(data_size), 'train')
            if not is_train:
                self.classes[c]['path'] = os.path.join(path, c, 'test')

            print 'reading the {} category'.format(c)

            files_list = os.listdir(self.classes[c]['path'])

            self.classes[c]['filenames'] = np.array(map(lambda f: os.path.join(self.classes[c]['path'], f), files_list))
            self.classes[c]['size'] = len(files_list)
            self.ndata += len(files_list)
            self.inv_classes[i] = c

        print 'fetched number of instances ', self.ndata

        self.n_features = data_size**3
        self.start = 0

        # mini-batch sized buffer
        self.Xbuf = self.be.iobuf(data_size**3)
        if not lshape is None:
            self.Xbuf.lshape = lshape

        assert self.ndata > self.be.bsz

        self.ybuf = None
        self.make_onehot = make_onehot
        if make_onehot:
            self.ybuf = self.be.iobuf(self.n_classes)
        else:
            self.ybuf = self.be.iobuf(1)

    @property
    def nbatches(self):
        return -((self.start - self.ndata) // self.be.bsz)

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        self.start = 0

    def __iter__(self):
        """
        Defines a generator that can be used to iterate over this dataset.

        Yields:
            tuple: The next minibatch. A minibatch includes both features and
            labels.
        """

        self.balance_data()
        permutation = np.random.permutation(self.ndata_balanced)

        for i1 in range(self.start, self.ndata, self.be.bsz):
            i2 = min(i1 + self.be.bsz, self.ndata)
            bsz = i2 - i1
            if i2 == self.ndata:
                self.start = self.be.bsz - bsz

            batch_indices = permutation[i1:i2]

            batch = self.read_batch(self.filenames_balanced[batch_indices])
            labels = self.labels_balanced[batch_indices]

            self.Xbuf[:, :bsz] = batch
            if self.be.bsz > bsz:
                batch_residue = self.read_batch(self.filenames_balanced[permutation[:self.be.bsz - bsz]])
                self.Xbuf[:, bsz:] = batch_residue

            if self.make_onehot:
                self.ybuf[:, :bsz] = self.be.onehot(self.be.array(labels.reshape(-1, 1)), axis=0)
                if self.be.bsz > bsz:
                    labels_residue = self.labels_balanced[permutation[:self.be.bsz - bsz]]
                    self.ybuf[:, bsz:] = self.be.onehot(self.be.array(labels_residue.reshape(-1, 1)), axis=0)
            else:
                self.ybuf[:, :bsz] = labels
                if self.be.bsz > bsz:
                    labels_residue = self.labels_balanced[permutation[:self.be.bsz - bsz]]
                    self.ybuf[:, bsz:] = labels_residue

            targets = self.ybuf if self.ybuf else self.Xbuf
            yield (self.Xbuf, targets)

    def balance_data(self):
        """
        balance categories with dramatically different number of data by
        duplicating the data of small categories so that each category has
        roughly the same number of data.
        """
        max_class = max(self.classes, key=lambda x: self.classes[x]['size'])
        max_class_size = self.classes[max_class]['size']
        n_batches = self.n_classes * max_class_size / self.be.bsz

        self.n_batches = n_batches

        # Reduce max_class_size until integer number of batches contains in max_class_size * self.n_classes
        while max_class_size * self.n_classes != n_batches * self.be.bsz:
            max_class_size = n_batches * self.be.bsz / self.n_classes
            n_batches = max_class_size * self.n_classes / self.be.bsz

        sample_size = max_class_size * self.n_classes
        self.ndata_balanced = sample_size

        self.filenames_balanced = []
        self.labels_balanced = np.zeros((sample_size, ), dtype=np.int32)
        for i, c in enumerate(self.classes):
            class_indices = np.arange(i * max_class_size, (i + 1) * max_class_size, dtype=int)
            self.labels_balanced[class_indices] = i
            if self.classes[c]['size'] >= max_class_size:
                #TODO: maybe use permutation
                self.filenames_balanced.append(self.classes[c]['filenames'][:max_class_size])
            else:
                ratio = max_class_size / self.classes[c]['size']
                self.filenames_balanced.append(np.tile(self.classes[c]['filenames'], ratio))
                residue = max_class_size - self.classes[c]['size'] * ratio

                permutation = np.random.permutation(self.classes[c]['size'])
                self.filenames_balanced.append(self.classes[c]['filenames'][permutation[:residue]])

        self.filenames_balanced = np.hstack(self.filenames_balanced)

    def read_batch(self, filenames, translation=False):
        """
        load the input batch from file(pre-computed mat files)
        Input:
            filenames (list of strings): list of filenames to read
            translation: 1 for translating the data by small move and 0 for nothing.
            Hopefully, introducing translation on data would train a better model.
        Returns:
            batch (ndarray): batch of size n_features x batch_size
        """

        batch = np.zeros((self.n_features, self.be.bsz), dtype=np.int8)
        for i in xrange(self.be.bsz):

            instance = loadmat(filenames[i])['instance']

            shifted = instance
            if translation:
                orientation = np.random.randint(7)
                move = 2
                shifted = np.zeros(self.lshape, dtype=np.int8)
                if orientation == 1:
                    shifted[:, :, :-move] = instance[:, :, move:]
                elif orientation == 2:
                    shifted[:, :, move:] = instance[:, :, :-move]
                elif orientation == 3:
                    shifted[:, :-move] = instance[:, move:]
                elif orientation == 4:
                    shifted[:, move:] = instance[:, :-move]
                elif orientation == 5:
                    shifted[:-move] = instance[move:]
                elif orientation == 6:
                    shifted[move:] = instance[:-move]

            batch[:, i] = shifted.reshape(-1)

        return batch

