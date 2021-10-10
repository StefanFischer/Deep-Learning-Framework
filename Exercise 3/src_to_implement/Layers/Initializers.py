"""
@description
Initialization is critical for non-convex optimization problems.
Depending on the application and network, different initialization strategies are required.
A popular initialization scheme is named Xavier or Glorot initialization.
Later an improved scheme specifically targeting ReLU activation functions was proposed by Kaiming He.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class Constant:
    def __init__(self, constant_value=0.1):
        """
        Object to realize the constant weight initialization.
        
        :param constant_value: Constant value used for weight initialization.
        """

        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Create the weight tensor with constant values.
        
        :param weights_shape: Desired shape of the weight tensor (fan_in, fan_out).
        :param fan_in: Input dimension of the weights (FC-layers) // [Nr. input channels * kernel height * kernel width] (Conv. layers).
        :param fan_out: Output dimension of the weights (FC-layers) // [Nr. output channels * kernel height * kernel width] (Conv. layers).
        :return: Initialized weight tensor of desired shape.
        """

        weight_tensor = np.full(weights_shape, self.constant_value)

        return weight_tensor


class UniformRandom:
    def __init__(self):
        """
        Object to realize the uniform ([0, 1)) weight initialization.
        """

        pass


    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Create the weight tensor with uniformly distributed values between [0, 1).
        
        :param weights_shape: Desired shape of the weight tensor (fan_in, fan_out).
        :param fan_in: Input dimension of the weights (FC-layers) // [Nr. input channels * kernel height * kernel width] (Conv. layers).
        :param fan_out: Output dimension of the weights (FC-layers) // [Nr. output channels * kernel height * kernel width] (Conv. layers).
        :return: Initialized weight tensor of desired shape.
        """

        weight_tensor = np.random.uniform(size=weights_shape)

        return weight_tensor


class Xavier:
    def __init__(self):
        """
        Object to realize the Xavier/ Glorot weight initialization.
        """

        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Create the weight tensor based on the Xavier/ Glorot initialization.
        Weights initialized by zero-mean Gaussian N(0, sigma), with sigma determined by size of input and output layer.
        
        :param weights_shape: Desired shape of the weight tensor (fan_in, fan_out).
        :param fan_in: Input dimension of the weights (FC-layers) // [Nr. input channels * kernel height * kernel width] (Conv. layers).
        :param fan_out: Output dimension of the weights (FC-layers) // [Nr. output channels * kernel height * kernel width] (Conv. layers).
        :return: Initialized weight tensor of desired shape.
        """

        sigma = np.sqrt(2 / (fan_out + fan_in))
        weight_tensor = np.random.normal(scale=sigma, size=weights_shape)

        return weight_tensor


class He:
    def __init__(self):
        """
        Object to realize the He weight initialization.
        """

        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Create the weight tensor based on the He initialization.
        Weights initialized by zero-mean Gaussian N(0, sigma), with sigma determined by size of previous layer only.

        :param weights_shape: Desired shape of the weight tensor (fan_in, fan_out).
        :param fan_in: Input dimension of the weights (FC-layers) // [Nr. input channels * kernel height * kernel width] (Conv. layers).
        :param fan_out: Output dimension of the weights (FC-layers) // [Nr. output channels * kernel height * kernel width] (Conv. layers).
        :return: Initialized weight tensor of desired shape.
        """

        sigma = np.sqrt(2 / fan_in)
        weight_tensor = np.random.normal(scale=sigma, size=weights_shape)

        return weight_tensor
