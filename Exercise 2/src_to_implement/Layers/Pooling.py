"""
@description
Pooling layers are typically used in conjunction with the convolutional layer.
They reduce the dimensionality of the input and therefore also decrease memory consumption.
Additionally, they reduce overfitting by introducing a degree of scale and translation invariance.
Max-pooling as the most common form of pooling is implemented here.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        """
        Constructor for a "Pooling Layer"-object.
        
        :param stride_shape: Single value or tuple. The latter allows for different strides in the spatial dimensions.
        :param pooling_shape: Dimension of the pooling neighborhood. (Typical choices: 2x2 or 3x3 neighborhoods)
        """

        # Store the stride dimensions
        self.stride_shape = stride_shape
        self.stride_y, self.stride_x = self.stride_shape

        # Store the pooling dimensions
        self.pooling_shape = pooling_shape
        self.pooling_y, self.pooling_x = self.pooling_shape

        # Store the previous input tensor.
        self.input_tensor_prev = None

        # Store the input tensor shape
        self.batch_size_input_prev, self.channel_size_input_prev, self.y_size_input_prev, self.x_size_input_prev = 0, 0, 0, 0


    def forward(self, input_tensor):
        """
        Forward pass to obtain the input_tensor for the next layer.
        Furthermore, store the necessary information for the backward pass.
        
        :param input_tensor: Input tensor for the current layer. Dimension: [B x C x Y x X]
        :return: Input tensor for the next layer.
        """

        # Store the previous input tensor for the backward pass.
        self.input_tensor_prev = input_tensor

        # Retrieve dimensions from the input shape.
        self.batch_size_input_prev, self.channel_size_input_prev, self.y_size_input_prev, self.x_size_input_prev = input_tensor.shape

        # Define the dimensions of the output by appling pooling.
        batch_size_new = self.batch_size_input_prev
        channel_size_new = self.channel_size_input_prev
        y_size_new = int(1 + (self.y_size_input_prev - self.pooling_y) / self.stride_y)
        x_size_new = int(1 + (self.x_size_input_prev - self.pooling_x) / self.stride_x)

        # Initialize the pooled output tensor
        input_tensor_new = np.zeros((batch_size_new, channel_size_new, y_size_new, x_size_new))

        for batch in range(batch_size_new):             # loop over the batches of the output volume
            for channel in range(channel_size_new):         # loop over the channels of the output volume
                for y in range(y_size_new):                     # loop over the vertical axis of the output volume
                    for x in range(x_size_new):                     # loop over the horizontal axis of the output volume

                        # Find the corners of the current "slice"
                        y_start = y * self.stride_y
                        y_end = y_start + self.pooling_y

                        x_start = x * self.stride_x
                        x_end = x_start + self.pooling_x

                        # Use the corners to define the current slice.
                        input_tensor_prev_slice = input_tensor[batch, channel, y_start:y_end, x_start:x_end]

                        # Compute the pooling operation on the slice.
                        input_tensor_new[batch, channel, y, x] = np.max(input_tensor_prev_slice)

        return input_tensor_new


    def backward(self, error_tensor):
        """
        Backward pass to obtain the error tensor for the previous layer.
        
        
        :param error_tensor: Error tensor of the current layer.
        :return: Error tensor for the previous layer.
        """

        # Retrieve dimensions the error tensor's shape
        batch_size_error, channel_size_error, y_size_error, x_size_error = error_tensor.shape

        # Initialize the previous error tensor with zeros
        error_tensor_prev = np.zeros(self.input_tensor_prev.shape)

        for batch in range(batch_size_error):           # loop over the batches of the error tensor
            for channel in range(channel_size_error):       # loop over the channels of the error tensor
                for y in range(y_size_error):                   # loop on the vertical axis of the error tensor
                    for x in range(x_size_error):                   # loop on the horizontal axis of the error tensor

                        # Find the corners of the current slice for the input_tensor_prev/ error_tensor_prev dimensions
                        y_start = y * self.stride_y
                        y_end = y_start + self.pooling_y

                        x_start = x * self.stride_x
                        x_end = x_start + self.pooling_x

                        # Define the current slice on the previous input tensor
                        input_tensor_prev_slice = self.input_tensor_prev[batch, channel, y_start:y_end, x_start:x_end]

                        # Create a mask from input_tensor_prev_slice to store where the maximum value was located.
                        mask = (input_tensor_prev_slice == np.max(input_tensor_prev_slice))

                        # Add to the previous error tensor at the location of the maximum the current error tensor value
                        error_tensor_prev[batch, channel, y_start:y_end, x_start:x_end] += (mask * error_tensor[batch, channel, y, x])

        return error_tensor_prev
