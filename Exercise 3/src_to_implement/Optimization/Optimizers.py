"""
@description
More advanced optimization schemes can increase speed of convergence.
We implement normal stochastic gradient descent, a pop-ular per-parameter adaptive scheme named Adam and
a common scheme improving stochastic gradient descent called momentum.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class Optimizer:
    def __init__(self):
        """
        Constructor for the class "Optimizer", which acts like the parent class for all optimizers of the network.
        """

        self.regularizer = None


    def add_regularizer(self, regularizer):
        """
        Add a regularizer.
        
        :param regularizer: Regularizer to be added.
        """

        self.regularizer = regularizer


class Sgd(Optimizer):
    """
    Implementation of the basic Stochastic Gradient Descent (SGD).
    """

    def __init__(self, learning_rate: float):
        """
        Create a "Stochastic Gradient Descent"-class object.

        :param learning_rate: Learning reate/ step size.
        """

        self.learning_rate = learning_rate

        # Inherit parent attributes
        super().__init__()


    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Return the updated weights according to the basic gradient descent update scheme.
        
        :param weight_tensor: Tensor containing all weights.
        :param gradient_tensor: Tensor containing the gradient of L with respect to the weights.
        :return: The updated weights.
        """

        # Apply regularization if available
        if self.regularizer:
            # Compute the regularized gradient of the weights
            regularized_gradient = self.regularizer.calculate_gradient(weight_tensor)

            # Compute the regularized weights
            weight_tensor = weight_tensor - self.learning_rate * regularized_gradient

        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):
    """
    Implementation of the "stochastic gradient descent with momentum"-method.
    """

    def __init__(self, learning_rate, momentum_rate):
        """
        Constructor for the "stochastic gradient descent with momentum"-method.
        
        :param learning_rate: Learning rate for gradient descent part.
        :param momentum_rate: Momentum rate for momentum part.
        """

        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        self.first_moment = 0

        # Inherit parent attributes
        super().__init__()


    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Update the weights.
        
        :param weight_tensor: Current weights.
        :param gradient_tensor: Gradient of the loss based on the weights.
        :return: Updated weights.
        """

        # Apply regularization if available
        if self.regularizer:
            # Compute the regularized gradient of the weights
            regularized_gradient = self.regularizer.calculate_gradient(weight_tensor)

            # Compute the regularized weights
            weight_tensor = weight_tensor - self.learning_rate * regularized_gradient

        self.first_moment = self.momentum_rate * self.first_moment - self.learning_rate * gradient_tensor

        return weight_tensor + self.first_moment


class Adam(Optimizer):
    """
    Implementation of the "Adaptive Moment Estimation (Adam)"-method.
    """

    def __init__(self, learning_rate, mu, rho):
        """
        Constructor for the "Adaptive Moment Estimation (Adam)"-method.
        
        :param learning_rate: Learning rate.
        :param mu: Hyperparameter for the calculation of the first moment (v).
        :param rho: Hyperparamter for the calculation of the second moment (r). 
        """

        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.k = 0
        self.first_moment = 0
        self.second_moment = 0
        self.epsilon = np.finfo(float).eps

        # Inherit parent attributes
        super().__init__()


    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Update the weights.
        
        :param weight_tensor: Current weights.
        :param gradient_tensor: Gradient of the loss based on the weights.
        :return: Updated weights.
        """

        # Apply regularization if available
        if self.regularizer:
            # Compute the regularized gradient of the weights
            regularized_gradient = self.regularizer.calculate_gradient(weight_tensor)

            # Compute the regularized weights
            weight_tensor = weight_tensor - self.learning_rate * regularized_gradient

        # Increment k to realize the k-th iteration.
        self.k += 1

        # Calculate the first moment v.
        self.first_moment = self.mu * self.first_moment + (1 - self.mu) * gradient_tensor

        # Calculate the second moment r.
        self.second_moment = self.rho * self.second_moment + (1 - self.rho) * gradient_tensor**2

        # Execute the bias correction.
        first_moment_corrected = self.first_moment / (1 - self.mu**self.k)
        second_moment_corrected = self.second_moment / (1 - self.rho**self.k)

        # Calculate the updated weights.
        updated_weights = weight_tensor - self.learning_rate * (first_moment_corrected / (np.sqrt(second_moment_corrected) + self.epsilon))

        return updated_weights

