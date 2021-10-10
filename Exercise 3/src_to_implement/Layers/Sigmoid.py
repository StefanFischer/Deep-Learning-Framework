"""
@description
Implementation of the common activation function: Sigmoid function.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class Sigmoid:
    def __init__(self):
        """
        Constructor for the Sigmoid activation function.
        """

        # Store the activated input to reuse it within the backward pass.
        self.activated_input = None


    def forward(self, input_tensor):
        """
        Forward pass for the sigmoid activation function.
        
        :param input_tensor: Input data.
        :return: The activated input data.
        """

        self.activated_input = 1 / (1 + np.exp(-input_tensor))

        return self.activated_input


    def backward(self, error_tensor):
        """
        Backward pass for the sigmoid activation function.
        
        :param error_tensor: Error tensor.
        :return: Gradient.
        """

        gradient = error_tensor * self.activated_input * (1 - self.activated_input)

        return gradient
