"""
@description
Dropout is a typical regularizer method for Deep Learning. It's most often used to regularize fully connected layers.
It enforces independent weights, reducing the effect of co-adaptation.
This layer has no adjustable parameters.
We choose to implement inverted dropout.

For further information, see: https://deepnotes.io/dropout

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np

# Import parent class
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        """
        Constructor for the Dropout-Layer.
        
        :param probability: Determining the fraction units to keep.
        """

        # Inherit parent attributes
        super().__init__()

        self.probability = probability

        self.mask = None


    def forward(self, input_tensor):
        """
        Forward pass to obtain the input tensor of the next layer.
        In Inverted Dropout we scale the output activation during training phase by the given probability,
        so that we can leave the network during testing phase untouched.
        
        :param input_tensor: Input tensor for the Dropout-Layer.
        :return: Input tensor for the next layer.
        """

        # If we are in the training phase we execute drop out
        if not self.testing_phase:
            # Create a mask to randomly define which values should be dropped and scale it with the probability
            self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape) / self.probability

            # Create the input tensor of the next layer by applying randomized drop out
            input_tensor = input_tensor * self.mask

        else:
            self.mask = self.probability

        return input_tensor

    def backward(self, error_tensor):
        """
        Backward pass to obtain the error tensor of the previous layer.
        The dropout layer has no learnable parameters, and doesn’t change the volume size of the output.
        So the backward pass is fairly simple.
        We simply back propagate the gradients through the neurons that were not killed off during the forward pass,
        as changing the output of the killed neurons doesn’t change the output, and thus their gradient is 0.
        
        :param error_tensor: Error tensor for the Dropout-Layer.
        :return: Error tensor for the previous layer.
        """

        # Back propagate the gradients for the neurons that weren't dropped out.
        error_tensor_prev = error_tensor * self.mask

        return error_tensor_prev
