"""
@description
Implementation of a Fully Connected Layer.
Layer oriented frameworks represent a higher level of abstraction to their users than graph oriented frameworks.
This approach limits flexibility but enables easy experimentation using conventional architectures.
The Fully Connected(FC) layer is the theoretic backbone of layer oriented architectures.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np

# Import parent class
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        """
        Construct a Fully Connected Layer object.
        
        :param input_size: Size of input layer.
        :param output_size: Size of output layer.
        """

        # Inherit parent attributes
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1, output_size)

        self.homogeneous_input_tensor = None

        self._optimizer = None
        self._gradient_weights = None


    @property
    def optimizer(self):
        """
        Getter for the protected member "optimizer". 
        """

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for the protected member "optimizer".
        """
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        """
        Getter for the protected member "_gradient_weights". 
        """

        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        """
        Setter for the protected member "gradient_weights".
        """
        self._gradient_weights = gradient_weights

    def initialize(self, weights_initializer, bias_initializer):
        """
        Initialize the weight tensor.
        
        :param weights_initializer: Object of the desired initializer.
        :param bias_initializer: Object of the desired initializer.
        """

        # Initialize the weights.
        weight_tensor_shape = (self.input_size, self.output_size)
        weights = weights_initializer.initialize(weight_tensor_shape, self.input_size, self.output_size)

        # Initialize the bias.
        bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)

        # Add the bias to the the weight tensor.
        self.weights = np.concatenate((weights, bias), axis=0)

    def forward(self, input_tensor):
        """
        Implementation the forward pass to obtain the input_tensor for the next layer.
        
        :param input_tensor: Matrix of dimensions: [input_size x batch_size] containing the input values.
        :return: Input tensor for next layer.
        """

        # Create a homogeneous representation of the input_tensor and
        # store it as a class variable to reuse it in the backward method
        batch_size = len(input_tensor)
        homogeneous_extension = np.ones((batch_size, 1))
        self.homogeneous_input_tensor = np.concatenate([input_tensor, homogeneous_extension], axis=1)

        return np.dot(self.homogeneous_input_tensor, self.weights)

    def backward(self, error_tensor):
        """
        Implementation of the backward pass to obtain the error tensor of the previous layer.
        
        :param error_tensor: Input error tensor.
        :return: Error tensor for previous layer.
        """

        # Calculate the Error tensor of the previous layer
        previous_error_tensor = np.dot(error_tensor, self.weights[:-1:].T)

        # Calculate the gradient with respect to the weights
        self.gradient_weights = np.dot(self.homogeneous_input_tensor.T, error_tensor)

        # Update the weights
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return previous_error_tensor
