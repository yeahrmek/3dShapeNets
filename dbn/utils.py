import numpy as np
from scipy.io import loadmat
import os


def read_data_list(base_path, classes, data_size, is_train=True, debug=False):
    """"
    read pre-computed volumetric filenames
    Input:
        base_path: root data folder
        classes: categories to be fetched
        data_size: the size of the volumetric representation
        is_train: select training data or testing data
        debug: true for debug.
    Reutrns:
        data_list: {'filename': [...], 'label': [...]}
    """

    if is_train:
        maxNum = 80 * 12
        filename_suffix = 'train'
    else:
        maxNum = 20 * 12
        filename_suffix = 'test'
    if debug:
        maxNum = 20

    n_classes = len(classes)
    data_list = dict()
    data_list['filename'] = []
    data_list['label'] = []
    for c in xrange(n_classes):
        print 'reading the %s category' % classes[c]
        category_path = os.path.join(base_path, classes[c], str(data_size), filename_suffix)

        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

        cat_idx = 0
        for f in files:
            if f[-3:] != 'mat':
                continue

            filename = os.path.join(category_path, f)
            cat_idx += 1

            data_list['filename'] += [filename]
            data_list['label'] += [c]
            if cat_idx == maxNum:
                break

        print 'fetched %d number of instances' % cat_idx

    return data_list


def read_batch(input_size, file_list, translation):
    """
    load the input batch from file(pre-computed mat files)
    Input:
        input_size: size of input image (e.g. [30, 30, 30])
        file_list: {'filename': [...], } dictionary (typically returned by read_data_list() function)
        translation: 1 for translating the data by small move and 0 for nothing.
                     Hopefully, introducing translation on data would train a better model.
    Returns:
        batch: array of size [len(file_list['filename'], input_size)]
    """

    batch_size = len(file_list['filename'])
    batch = np.zeros([batch_size] + [s for s in input_size])
    for i in xrange(batch_size):
        data = loadmat(file_list['filename'][i])
        instance = data['instance']
        if translation:
            orient = np.random.randint(7)
            move = 2
            shifted = zeros(instance.shape)
            if orient == 0:   # 'z'
                shifted[:, :, :-move] = instance[:, :, move:]

            elif orient == 1: # 'z'
                shifted[:, :, move:] = instance[:, :, :-move]

            elif orient == 2: # 'y'
                shifted[:, :-move, :] = instance[:, move:, :]

            elif orient == 3: # 'y'
                shifted[:, move:, :] = instance[:, :-move, :]

            elif orient == 4: # 'x'
                shifted[:-move, :, :] = instance[move:, :, :]

            elif orient == 5: # 'x'
                shifted[move:, :, :] = instance[:-move, :, :]

            elif orient == 6: #' none'
                shifted = instance
        else:
            shifted = instance

        batch[i, :, :, :] = shifted

    return batch