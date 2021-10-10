"""
@description
The SoftMax activation function is used to transform the logits (the output of the network)
into a probability distribution.
Therefore, SoftMax is typically used for classification tasks.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class SoftMax:
    def __init__(self):
        self.predictions = None

    def forward(self, input_tensor):
        """
        Estimate the class probabilities for each element of the batch (rows of input tensor).
        For the formulas, see 1_BasicFramework.pdf, page 12f.
        
        :param input_tensor: Input values.
        :return: Estimated class probabilities.
        """

        # Increase the numerical stability by shift the input values xk -> xk - max(x)
        x_max = np.max(input_tensor)

        stabilized_input_tensor = input_tensor - x_max

        # Calculate the activation(prediction) y^ for every element of the batch
        exp_input_tensor = np.exp(stabilized_input_tensor)

        sum_exp_xN = np.sum(exp_input_tensor, axis=1, keepdims=True)

        self.predictions = exp_input_tensor / sum_exp_xN

        return self.predictions

    def backward(self, error_tensor):
        """
        Backward pass for calculating the error tensor of the previous layer.
        For the formula, see 1_BasicFramework.pdf, page 14.
        
        :param error_tensor: Error tensor of the current layer.
        :return: Error tensor of the previous layer.
        """

        error_prediction_products = np.multiply(error_tensor, self.predictions)

        sums_error_prediction_products = np.sum(error_prediction_products, axis=1, keepdims=True)

        prev_error_tensor = self.predictions * (error_tensor - sums_error_prediction_products)

        return prev_error_tensor
