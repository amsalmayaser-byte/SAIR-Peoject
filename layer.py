import numpy as np

class Dense:
    def __init__(self, input_size, output_size, activation=None,dropout_rate=0.0,l2_lambda=0.01):
        self.activation = activation
        self.dropout_rate =dropout_rate
        self.l2_lambda = l2_lambda
        self.mask = None
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
        if training and self.dropout_rate > 0:
        self.mask = (np.random.rand(*self.output.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
        self.output *= self.mask
        return self.output

    def backward(self, grad_batch):
        if hasattr(self, 'dropout_rate') and self.dropout_rate > 0 and self.mask is not None:
            grad_batch = grad_batch * self.mask
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
