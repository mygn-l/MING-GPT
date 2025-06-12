import numpy as np
from Feedforward import Feedforward

class Multilayer_Feedforward:
    def __init__(self, num_layers, dims):
        self.layers = [Feedforward(dims) for i in range(num_layers)]
        self.dims = dims
        self.num_layers = num_layers

    def forward_train(self, input_vectors):
        output_vectors = np.empty(input_vectors.shape)
        for i in range(self.num_layers):
            output_vectors[i, :] = self.layers[i].forward_train(input_vectors[i, :])
        return output_vectors

    def backward(self, dC_dY):
        dC_dX = np.empty((self.num_layers, self.dims[0]))
        for i in range(self.num_layers):
            dC_dX[i, :] = self.layers[i].backward(dC_dY[i, :])
        return dC_dX
