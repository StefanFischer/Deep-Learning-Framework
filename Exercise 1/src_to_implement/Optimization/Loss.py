"""
@description
The cross entropy Loss is often used in classification task, typically in conjunction with SoftMax (or Sigmoid).

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        """
        Compute the loss value according to the "Cross Entropy Loss" formula accumulated over the batch.
        
        :param input_tensor: Activation(prediction) \hat{y} for every element of the batch of size B.
        :param label_tensor: Tensor containing the labels 0 and 1.
        :return: Cross entropy loss.
        """

        self.input_tensor = input_tensor

        loss = np.sum(np.where(label_tensor == 1, -np.log(input_tensor + np.finfo(float).eps), [0]))

        return loss

    def backward(self, label_tensor):
        """
        Backward pass to return the error tensor for the previous layer.
        
        :param label_tensor: Tensor containing the labels 0 and 1.
        :return: Error tensor for the previous layer.
        """

        error_tensor = - (label_tensor / self.input_tensor)

        return error_tensor
