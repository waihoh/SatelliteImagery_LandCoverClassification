import numpy as np
from tqdm import tqdm


def classcount(loader, num_class=7):
    """
    Calculates the weights to be used for the weighted cross entropy by counting the pixel number of each class in the train dataset
    :param loader: train dataset
    :param num_class: 7 classes
    :return: weights
    """
    n_train = len(loader)

    class_weight = np.array([0.0]*num_class)

    with tqdm(total=n_train, desc='Class Count Assessment', unit='batch', disable=False, leave=True) as pbar:
        for batch in loader:
            imgs, true_mask = batch['image'], batch['mask']
            (unique, counts) = np.unique(ar=true_mask, return_counts=True)
            frequencies = np.asarray(a=(unique, counts))

            for i in range(frequencies.shape[1]):
                class_weight[frequencies[0, i]] += frequencies[1, i]  # TODO: check values
            pbar.update()

    class_weight = class_weight[:-1].min()/class_weight
    class_weight[-1] = 0

    return class_weight
