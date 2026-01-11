import numpy as np

class Dense:
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros((1, output_size))

    def forward(self, X_batch):
        self.input = X_batch
        self.z = np.dot(self.input, self.weights) + self.bias
        if self.activation == 'tanh': self.output = np.tanh(self.z)
        elif self.activation == 'sigmoid': self.output = 1 / (1 + np.exp(-self.z))
        elif self.activation == 'relu': self.output = np.maximum(0, self.z)
        else: self.output = self.z
        return self.output

    def backward(self, grad_batch):
        if self.activation == 'tanh': dz = grad_batch * (1 - self.output ** 2)
        elif self.activation == 'sigmoid': dz = grad_batch * (self.output * (1 - self.output))
        elif self.activation == 'relu':
            dz = np.array(grad_batch, copy=True)
            dz[self.z <= 0] = 0
        else: dz = grad_batch
        self.weights_gradient = np.dot(self.input.T, dz)
        self.bias_gradient = np.sum(dz, axis=0, keepdims=True)
        input_gradient = np.dot(dz, self.weights.T)
        return input_gradient

    def update_weights(self, optimizer):
        optimizer.update(self)