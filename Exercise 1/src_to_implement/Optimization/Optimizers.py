"""
@description
This is the implementation of the basic Stochastic Gradient Descent (SGD).

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""


class Sgd:
    def __init__(self, learning_rate: float):
        """
        Create a "Stochastic Gradient Descent"-class object.

        :param learning_rate: Learning reate/ step size.
        """

        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Return the updated weights according to the basic gradient descent update scheme.
        
        :param weight_tensor: Tensor containing all weights.
        :param gradient_tensor: Tensor containing the gradient of L with respect to the weights.
        :return: The updated weights.
        """

        return weight_tensor - self.learning_rate * gradient_tensor
