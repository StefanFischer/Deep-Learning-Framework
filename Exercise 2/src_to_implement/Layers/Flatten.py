"""
@description
Flatten layers reshapes the multi-dimensional input to a one dimensional feature vector.
This is useful especially when connecting a convolutional or pooling layer with a fully connected layer.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class Flatten:
    def __init__(self):
        """
        Constructor for a flatten layer object.
        """

        self.batch_size = 0
        self.nr_input_channels = 0
        self.input_size_y = 0
        self.input_size_x = 0

    def forward(self, input_tensor):
        """
        Flatten the input tensor for each batch to a 1D linear representation.
        
        :param input_tensor: Originally shaped input tensor.
        :return: Linearized batch-wise representation of the input tensor.
        """

        original_shape = input_tensor.shape

        # Extract the batch size out of the input tensor
        self.batch_size = original_shape[0]

        # Extract the dimensions [C x Y x X] of the individual layers out of the input tensor.
        self.nr_input_channels = original_shape[1]
        self.input_size_y = original_shape[2]
        self.input_size_x = original_shape[3]

        layer_size_1d = np.product([self.nr_input_channels, self.input_size_y, self.input_size_x])

        # Flatten the whole input tensor into a 1D-representation
        flattened_input_tensor = np.ndarray.flatten(input_tensor)

        # Reshape the representation to meet the different batches again
        flattened_batch_representation = np.reshape(flattened_input_tensor, (self.batch_size, layer_size_1d))

        return flattened_batch_representation

    def backward(self, error_tensor):
        """
        Reshape the error tensor to the original shape dimensions.
        
        :param error_tensor: 1D linear representation of the error tensor.
        :return: Reshaped representation of the error tensor.
        """

        # Reshape the error tensor to [B x C x Y x X]
        reshaped_error_tensor = np.reshape(error_tensor, (self.batch_size, self.nr_input_channels, self.input_size_y, self.input_size_x))

        return reshaped_error_tensor

