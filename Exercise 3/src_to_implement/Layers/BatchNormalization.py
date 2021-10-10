"""
@description
Batch Normalization is a regularization technique which is conceptually very well known in
Machine Learning but specially adapted to Deep Learning.

Batch normalization deals with the problem of poorly initialization of neural networks.
It can be interpreted as doing preprocessing at every layer of the network.
It forces the activations in a network to take on a unit gaussian distribution at the beginning of the training.
This ensures that all neurons have about the same output distribution in the network and improves the rate of convergence

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np

# Import parent class
from Layers.Base import BaseLayer
from Layers import Helpers


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        """
        Constructor for the "batch normalization"-layer.
        
        :param channels: Number of channels within the batch.
        """

        # Inherit parent attributes
        super().__init__()

        self.channels = channels

        # Weights and bias for this layer
        self.weights, self.bias = None, None
        self.initialize()

        self.epsilon = 1e-15  # Has to be smaller than 1e-10

        # Variance, mean and standard deviation for the forward pass for both, train and test phase
        self.var_input_tensor = None

        self.mean_batch, self.std_batch = None, None

        self.mean_moving_average, self.var_moving_average = None, None

        # Input tensor and error tensor
        self.input_tensor_prev, self.input_tensor, self.input_tensor_normalized, self.error_tensor = None, None, None, None

        # Reformatting dimensions
        self.B, self.H, self.M, self.N = None, None, None, None

        # Optimizer for updating the weights and the bias within the backward pass
        self.optimizer = None

        # Gradients
        self.gradient_weights = None
        self.gradient_bias = None


    @property
    def optimizer(self):
        """
        Getter for the protected member "optimizer". 
        """

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for the protected member "optimizer".
        """

        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        """
        Getter for the protected member "gradient_weights". 
        """

        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        """
        Setter for the protected member "gradient_weights".
        """

        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        """
        Getter for the protected member "gradient_bias". 
        """

        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        """
        Setter for the protected member "gradient_bias".
        """

        self._gradient_bias = gradient_bias


    def initialize(self, weights_initializer=None, bias_initializer=None):
        """
        Initialize the weights "gamma" with ones and the bias "beta" with zeros.
        Since we do not want the weights and bias to have an impact at the beginning of the training.
        Ignore any assigned initializer!
        """

        # Initalize the weights as a vector containing only ones.
        self.weights = np.ones((1, self.channels))

        # Initialize the bias as a vector containing only zeros.
        self.bias = np.zeros((1, self.channels))


    def reformat(self, tensor):
        """
        Depending on the shape of the tensor, the method reformats the image-like tensor (with 4 dimension) into
        its vector-like variant (with 2 dimensions), and the same method reformats the vector-like tensor into
        its image-like tensor variant.
        This is necessary for convolutional batch normalization.
        
        :param tensor: Image- or vector-like variant.
        :return: Reformatted image- or vector-like variant depending on the variant of the input tensor.
        """

        # Reformat an image-like tensor [B x H x M x N] into a vector-like tensor [B * M * N x H]
        if len(tensor.shape) == 4:
            self.B, self.H, self.M, self.N = tensor.shape
            tensor = np.reshape(tensor, (self.B, self.H, self.M * self.N))
            tensor = np.transpose(tensor, axes=(0, 2, 1))
            tensor = np.reshape(tensor, (self.B * self.M * self.N, self.H))

        # Reformat a vector-like tensor [B * M * N x H] into an image-like tensor [B x H x M x N]
        else:
            tensor = np.reshape(tensor, (self.B, self.M * self.N, self.H))
            tensor = np.transpose(tensor, axes=(0, 2, 1))
            tensor = np.reshape(tensor, (self.B, self.H, self.M, self.N))

        return tensor


    def forward(self, input_tensor):
        """
        Forward pass of the "batch normalization"-layer.
        
        :param input_tensor: Input data.
        :return: Input tensor for the next layer.
        """

        # ==============================================================================================================
        # Step 1: Reformat the input into a vectorized (2D) representation, if it is a batch of images (4D).
        #         This allows to use the same code for Fully Connected as well as Convolutional Neural Networks.
        #         Also store the input tensor for reuse within the backward pass.
        # ==============================================================================================================
        self.input_tensor_prev, self.input_tensor = input_tensor, input_tensor

        if len(self.input_tensor_prev.shape) == 4:
            self.input_tensor = self.reformat(input_tensor)

        # ==============================================================================================================
        # Step 2: Execute the batch normalization for the training phase.
        # ==============================================================================================================

        # If we are in the training phase we can sample over the batch and therefore use mean_ and std_batch
        if not self.testing_phase:
            # ==========================================================================================================
            # Step 2.1: Calculate the mean and standard deviation channelwise over the batch of the input data
            # ==========================================================================================================

            # Calculate the mean for each channel for each data element over the whole batch
            self.mean_batch = np.mean(self.input_tensor, axis=0)

            # Calculate the standard deviation for each channel for each data element over the whole batch
            self.std_batch = np.std(self.input_tensor, axis=0)

            # ==========================================================================================================
            # Step 2.2: Execute the batch normalization on the input data with the calculated mean, variance and std.
            # ==========================================================================================================

            # Calculate the centered input data
            input_tensor_centered = self.input_tensor - self.mean_batch

            # Calculate the normalized input data
            self.input_tensor_normalized = input_tensor_centered / np.sqrt(np.square(self.std_batch) + self.epsilon)

            # ==========================================================================================================
            # Step 2.3: Calculate the moving average to get the mean, variance and standard deviation for the test phase
            # ==========================================================================================================

            alpha = 0.8

            # For the first batch used for training, set the moving average to the batch respective mean and variance.
            if self.mean_moving_average is None:
                self.mean_moving_average = self.mean_batch
                self.var_moving_average = self.std_batch

            # For the next batches apply moving average for the mean as well as variance calculation
            else:
                # Apply moving average for the mean as well as variance calculation
                self.mean_moving_average = alpha * self.mean_moving_average + (1 - alpha) * self.mean_batch
                self.var_moving_average = alpha * self.var_moving_average + (1 - alpha) * self.std_batch

        # ==============================================================================================================
        # Step 3: Execute the batch normalization for the test phase.
        # ==============================================================================================================

        # If we are in the test phase we use the results of the moving average to get the mean and standard deviation.
        else:
            # ==========================================================================================================
            # Step 3.1: Execute the batch normalization on the input data with the calculated mean and variance.
            # ==========================================================================================================

            # Calculate the standard deviation based on the variance
            std_moving_average = np.sqrt(np.square(self.var_moving_average) + self.epsilon)

            # Calculate the centered input data
            input_tensor_centered = self.input_tensor - self.mean_moving_average

            # Calculate the normalized input data
            self.input_tensor_normalized = input_tensor_centered / std_moving_average

        # ==============================================================================================================
        # Step 4: Calculate the input tensor for the next layer using the weights and the bias.
        # ==============================================================================================================

        # Apply weights and bias onto our normalized input tensor
        output = self.weights * self.input_tensor_normalized + self.bias

        # ==============================================================================================================
        # Step 5: Reverse step 1, if it was executed, to obtain the correct image-wise batch representation of the data.
        # ==============================================================================================================

        if len(self.input_tensor_prev.shape) == 4:
            output = self.reformat(output)

        # ==============================================================================================================
        # Step 6: Calculate the variance of the 2D input tensor over the batch to reuse it in the backward pass.
        # ==============================================================================================================

        self.var_input_tensor = np.var(self.input_tensor, axis=0)

        # ==============================================================================================================
        # Step 7: Return the input tensor for the next layer with the correct dimensions for a (FC) or (CNN) network.
        # ==============================================================================================================

        return output


    def backward(self, error_tensor):
        """
        Backward pass of the "batch normalization"-layer.
        
        :param error_tensor: Error tensor of the current layer.
        :return: Error tensor of the previous layer.
        """

        self.error_tensor = error_tensor

        # ==============================================================================================================
        # Step 1: Reformat the input into a vectorized (2D) representation, if it is a batch of images (4D).
        #         This allows to use the same code for Fully Connected as well as Convolutional Neural Networks.
        # ==============================================================================================================
        if len(self.error_tensor.shape) == 4:
            error_tensor = self.reformat(error_tensor)

        # ==============================================================================================================
        # Step 2: Calculate the gradient w.r.t. the weights, the bias and the input.
        # ==============================================================================================================

        # Calculate the gradient with respect to the weights as the sum over the batch: sum(error * input)
        self.gradient_weights = np.sum(error_tensor * self.input_tensor_normalized, axis=0)[np.newaxis]

        # Calculate the gradient with respect to the bias as the sum over the batch: sum(error)
        self.gradient_bias = np.sum(error_tensor, axis=0)[np.newaxis]

        # Calculate the gradient with respect to the input
        gradient_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean_batch, self.var_input_tensor)

        # ==============================================================================================================
        # Step 3: Optimize the weights and the bias of the "batch normalization"-layer if an optimizer was set.
        # ==============================================================================================================

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        # ==============================================================================================================
        # Step 4: If the input tensor was 4D, then reshape the gradient of the input in the respective way
        # ==============================================================================================================

        if len(self.input_tensor_prev.shape) == 4:
            gradient_input = self.reformat(gradient_input)

        return gradient_input
