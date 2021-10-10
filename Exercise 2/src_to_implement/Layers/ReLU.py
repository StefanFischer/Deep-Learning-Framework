"""
@description
The Rectified Linear Unit is the standard activation function in Deep Learning nowadays.
It has revolutionized Neural Networks because it reduces the effect of the "vanishing gradient" problem.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np

class ReLU:
    def __init__(self):
        """
        Construct a ReLU-function object.
        """

        self.input_tensor = None

    def forward(self, input_tensor):
        """
        Forward pass for using the ReLU(Rectified Linear Unit)-function.
        
        :param input_tensor: Input tensor for the current layer.
        :return: Input tensor for the next layer.
        """

        self.input_tensor = input_tensor

        relu_passed_input_tensor = np.maximum(0, input_tensor)

        return relu_passed_input_tensor

    def backward(self, error_tensor):
        """
        Backward pass for using the ReLU(Rectified Linear Unit)-function.
        
        :param error_tensor: Error tensor of the current layer.
        :return: Error tensor of the previous layer.
        """

        relu_passed_error_tensor = np.where(self.input_tensor > 0, error_tensor, [0])

        return relu_passed_error_tensor
