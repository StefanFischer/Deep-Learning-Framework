"""
@description
Implementation of the common activation function: The hyperbolic tangent (Tangens Hyperbolicus).

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class TanH:
    def __init__(self):
        """
        Constructor for the TanH activation function.
        """

        # Store the activated input to reuse it within the backward pass.
        self.activated_input = None


    def forward(self, input_tensor):
        """
        Forward pass for the TanH activation function.

        :param input_tensor: Input data.
        :return: The activated input data.
        """

        self.activated_input = np.tanh(input_tensor)

        return self.activated_input


    def backward(self, error_tensor):
        """
        Backward pass for the TanH activation function.

        :param error_tensor: Error tensor.
        :return: Gradient.
        """

        gradient = error_tensor * (1 - np.square(self.activated_input))

        return gradient


