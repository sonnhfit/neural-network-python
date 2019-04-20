from .layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape;
        self.output_shape = output_shape;
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5;
        self.bias = np.random.rand(1, output_shape[1]) - 0.5;

    def forward_propagation(self, input):
        self.input = input;
        self.output = np.dot(self.input, self.weights) + self.bias;
        return self.output;

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T);
        dWeights = np.dot(self.input.T, output_error);
        self.weights -= learning_rate * dWeights;
        self.bias -= learning_rate * output_error;
        return input_error;
