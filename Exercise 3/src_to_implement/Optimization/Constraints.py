"""
@description
Equality and inequality constraints on the norm of the weights are well known in the field of Machine Learning.
They enforce the prior assumption on the model of small weights in case of L2-regularization
and sparsity in case of L1-regularization.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class L2_Regularizer:
    def __init__(self, alpha):
        """
        Constructor for a L2-Regularization object.
        
        :param alpha: Regularization weight.
        """

        self.alpha = alpha

    def calculate_gradient(self, weights):
        """
        Calculate a (sub-)gradient on the weights needed for the optimizer.
        
        :param weights: Weights.
        :return: Regularized gradient of the weights.
        """

        return self.alpha * weights

    def norm(self, weights):
        """
        Calculate the norm enhanced loss.
        
        :param weights: Weights.
        :return: Squared L2-Norm of the weights.
        """

        return self.alpha * np.sum(weights**2)


class L1_Regularizer:
    def __init__(self, alpha):
        """
        Constructor for a L1-Regularization object.

        :param alpha: Regularization weight.
        """

        self.alpha = alpha

    def calculate_gradient(self, weights):
        """
        Calculate a (sub-)gradient on the weights needed for the optimizer.

        :param weights: Weights.
        :return: Regularized gradient of the weights.
        """

        return self.alpha * np.sign(weights)

    def norm(self, weights):
        """
        Calculate the norm enhanced loss.

        :param weights: Weights.
        :return: L1-Norm of the weights.
        """

        return self.alpha * np.sum(np.abs(weights))
