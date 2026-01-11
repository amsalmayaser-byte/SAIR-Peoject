import numpy as np

class BatchGenerator:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.indices = np.arange(self.n_samples)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]
