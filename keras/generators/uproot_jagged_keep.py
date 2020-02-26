from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
import numpy as np
import uproot

from generators.utils import to_dense

class UprootJaggedSequence(keras.utils.Sequence):
    def __init__(self, paths, batch_size, format='xn', features=None, n_vert_max=256, y_dtype=np.int, y_features=None, dataset_name='events'):
        super(UprootJaggedSequence, self).__init__()
        
        self.batch_size = batch_size
        self.n_vert_max = n_vert_max

        self.format = format

        n_events = 0
        for data in uproot.iterate(paths, dataset_name, ['n'], namedecode='ascii'):
            n_events += data['n'].shape[0]
    
        self.n_steps = n_events // batch_size
    
        self.n = np.empty((n_events,), dtype=np.int)

        if y_features is None:
            shape = (n_events,)
        else:
            shape = (n_events, len(y_features))

        self.y = np.empty(shape, dtype=y_dtype)
    
        start = 0
        for data in uproot.iterate(paths, dataset_name, ['n', 'y'], namedecode='ascii'):
            end = start + data['n'].shape[0]
            self.n[start:end] = data['n']
            if y_features is not None:
                self.y[start:end] = data['y'][:, y_features]
            else:
                self.y[start:end] = data['y']
            start = end
    
        if features is None:
            for data in uproot.iterate(paths, dataset_name, ['x'], namedecode='ascii', entrysteps=1):
                nfeat = data['x'].content.shape[1]
                break
        else:
            nfeat = len(features)
    
        nhits_total = np.sum(self.n)
        self.xcont = np.empty((nhits_total, nfeat), dtype=np.float)
    
        cstart = 0
        for data in uproot.iterate(paths, dataset_name, ['x'], namedecode='ascii'):
            x = data['x'].content
            cend = cstart + x.shape[0]
            if features is None:
                self.xcont[cstart:cend] = x
            else:
                self.xcont[cstart:cend] = x[:, features]
    
            cstart = cend

        self.xpos = np.empty((self.n_steps + 1,), dtype=np.int)
        self.set_xpos()

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):
        start = self.batch_size * index
        end = start + self.batch_size
        nbatch = self.n[start:end]

        xstart = self.xpos[index]
        xend = self.xpos[index + 1]
        x = to_dense(nbatch, self.xcont[xstart:xend], n_vert_max=self.n_vert_max)

        if self.format == 'xn':
            return [x, nbatch], self.y[start:end]
        elif self.format == 'xen':
            return [x[:, :, :3], x[:, :, 3], nbatch], self.y[start:end]

    def set_xpos(self):
        xpos = 0
        for index in range(self.n_steps):
            self.xpos[index] = xpos
            start = self.batch_size * index
            end = start + self.batch_size
            xpos += np.sum(self.n[start:end])

        self.xpos[self.n_steps] = xpos

        

def make_generator(paths, batch_size, format='xn', features=None, n_vert_max=256, y_dtype=np.int, y_features=None, dataset_name='events'):
    sequence = UprootJaggedSequence(paths, batch_size, format, features, n_vert_max, y_dtype, y_features, dataset_name)

    return sequence, None


def make_dataset(path, format='xn', features=None, n_vert_max=256, y_features=None, n_sample=None, dataset_name='events'):
    data = uproot.open(path)[dataset_name].arrays(['x', 'n', 'y'], namedecode='ascii')

    n = data['n']

    x = to_dense(n, data['x'].content, n_vert_max=n_vert_max)
    if features is not None:
        x = x[:, features]

    if format == 'xen':
        e = x[:, :, 3]
        x = x[:, :, :3]

    if y_features is None:
        y = data['y']
    else:
        y = data['y'][:, y_features]

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

    return inputs, truth, True


if __name__ == '__main__':
    import sys

    path = sys.argv[1]

    sequence, n_steps = make_generator(path, 1, n_vert_max=1024, y_features=[0])

    print(n_steps, 'steps')
    print(sequence[0])
