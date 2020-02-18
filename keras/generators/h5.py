import random
import h5py
import numpy as np
def make_generator(paths, batch_size, format='xn', features=None, n_vert_max=256, y_dtype=np.int, y_features=None, dataset_name='events'):
    paths = list(paths)

    n_steps = 0
    for path in paths:
        with h5py.File(path, 'r') as f:
            n_steps += f['n'].shape[0] // batch_size

    def get_event():
        while True:
            random.shuffle(paths)

            for path in paths:
                with h5py.File(path, 'r') as f:
                    start = 0
                    end = batch_size
                    while start < f['n'].shape[0] - 1:
                        if features is None:
                            x = f['x'][start:end]
                        else:
                            x = f['x'][start:end][ :, features]

                        if y_features is None:
                            y = f['y'][start:end]
                        else:
                            y = f['y'][start:end][:, y_features]

                        yield x, f['n'][start:end], y

                        start = end
                        end += batch_size

    return get_event(), n_steps


def make_dataset(path, format='xn', features=None, n_vert_max=256, y_features=None, n_sample=None, dataset_name='events'):
    f = h5py.File(path)
    if features is None:
        x = f['x']
    else:
        x = f['x'][:, features]

    if format == 'xen':
        e = x[:, :, 3]
        x = x[:, :, :3]

    if y_features is None:
        y = f['y']
    else:
        y = f['y'][:, y_features]

    n = f['n']

    if n_sample is not None:
        x = x[:n_sample]
        n = n[:n_sample]
        y = y[:n_sample]
        if format == 'xen':
            e = e[:n_sample]

    if format == 'xn':
        inputs = [x, n]
    elif format == 'xen':
        inputs = [x, e, n]
    truth = y

    return inputs, truth, 'batch'

