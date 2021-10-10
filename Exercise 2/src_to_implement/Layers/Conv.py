"""
@description
While fully connected layers are theoretically well suited to approximate any function,
they struggle to efficiently classify images due to extensive memory consumption and overfitting.
Using convolutional layers,
these problems can be circumvented by restricting the layer's pa-rameters to local receptive fields.

@questions:
Why do I need to extract the middle channel in the forward pass?

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np
from scipy import signal


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """
        Constructor for a "Convolutional Layer"-object.
        
        :param stride_shape: Single value or tuple. The latter allows for different strides in the spatial dimensions.
        :param convolution_shape: Determines whether this object provides a 1D [c, m] or a 2D [c, m, n] convolution layer.
        :param num_kernels: Number of filter kernels.
        """

        self.stride_shape = stride_shape
        self.stride_y, self.stride_x = 0, 0

        self.convolution_shape = convolution_shape
        self.input_tensor_shape = None
        self.input_tensor = None

        self.num_kernels = num_kernels

        # Initialize the filter uniformly random in the range [0, 1) as tensor of size: [Nr_kernels x C x M (x N)].
        self.weights = np.random.uniform(size=(self.num_kernels, *self.convolution_shape))

        # Initialize the bias uniformly random in the range [0, 1) for each kernel as a single value.
        self.bias = np.random.uniform(size=self.num_kernels)

        self.gradient_weights = None  # Gradient with respect to the weights.
        self.gradient_bias = None  # Gradient with respect to the bias.

        self.optimizer = None  # Optimizer of this layer.

    @property
    def gradient_weights(self):
        """
        Getter for the protected member "gradient_weights". 
        """

        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weights):
        """
        Setter for the protected member "gradient_weights".
        """

        self._gradient_weights = weights

    @property
    def gradient_bias(self):
        """
        Getter for the protected member "gradient_bias". 
        """

        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, bias):
        """
        Setter for the protected member "gradient_bias".
        """

        self._gradient_bias = bias

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


    def forward(self, input_tensor):
        """
        Forward pass to obtain the input_tensor for the next layer.
        
        :param input_tensor: Input tensor of current layer 1D:[B x C x Y] // 2D:[B x C x Y x X].
        :return: Input tensor for next layer.
        """
        # Store the input tensor and its shape to use it within the backward method
        self.input_tensor = input_tensor
        self.input_tensor_shape = input_tensor.shape

        # Check if number of image channels matches the filter depth.
        if self.input_tensor_shape[1] != self.weights.shape[1]:
            print("Error: Number of channels in both input and filter must match.")
            raise SystemExit

        # Define the stride parameter(s)
        if len(self.stride_shape) == 1:
            self.stride_y, self.stride_x = self.stride_shape[0], self.stride_shape[0]
        else:
            self.stride_y, self.stride_x = self.stride_shape

        # Extract the number of channels
        nr_channels = self.convolution_shape[0]

        # Store all feature maps in a batch representation (4D) of 3D feature maps
        all_feature_maps = None

        # Generate for each input 1D signal or 2D image the corresponding feature map and stack them up
        for image in self.input_tensor:
            # Store all convolutions to the one current 1D signal / 2D image in a feature map (2D / 3D numpy.array([]))
            feature_map = None

            for filter_kernel, bias in zip(self.weights, self.bias):
                # Execute the convolution of the current 1D signal / 2D image with the current kernel
                convolved_image = signal.correlate(image, filter_kernel, mode='same')

                # Extract convolution of the center channel
                convolved_image_center_channel = convolved_image[nr_channels // 2]

                # Execute the downsampling with the provided strip size for the 1D signal / 2D image
                if len(self.convolution_shape) == 2:
                    strided_image = convolved_image_center_channel[::self.stride_y]

                else:
                    strided_image = convolved_image_center_channel[::self.stride_y, ::self.stride_x]

                # Add bias to the strided 1D signal/ 2D image
                strided_image += bias

                # Add the strided 1D signal / 2D image to a stack to create the feature map
                if feature_map is None:
                    # Transform to a higher dimensional representation, to be able to stack all strided images together
                    feature_map = np.array([strided_image])

                else:
                    # Add the new strided 1D signal / 2D image to the stack
                    feature_map = np.concatenate((feature_map, [strided_image]))

            # Add the created feature map to a stack to get a batch representation of all feature maps
            if all_feature_maps is None:
                # Transform first feature map to a batch representation, to be able to stack all feature maps together
                all_feature_maps = np.array([feature_map])

            else:
                # Add the new generated feature map to the stack of feature maps
                all_feature_maps = np.concatenate((all_feature_maps, [feature_map]))

        return all_feature_maps


    def backward(self, error_tensor):
        """
        Backward pass to obtain the error tensor for the previous layer.
        
        :param error_tensor: Error tensor of the current layer.
        :return: Error tensor of the previous layer.
        """

        """
        # Calculate the Error tensor of the previous layer
        previous_error_tensor = np.dot(error_tensor, self.weights[:-1:].T)

        # Calculate the gradient with respect to the weights
        self.gradient_weights = np.dot(self.homogeneous_input_tensor.T, error_tensor)

        # Update the weights
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        """

        # ==============================================================================================================
        # Step 1: Adapt the error tensor to match the image dimensions of the input tensor by adding 0 where necessary
        # ==============================================================================================================
        # Create batch with 0 everywhere
        error_tensor_adapted = np.zeros((error_tensor.shape[0], error_tensor.shape[1], *self.input_tensor_shape[2:]))

        # Write the values of the error tensor at the specific positions based on the used stride parameters
        if len(self.convolution_shape) == 2:
            # In case we have a batch of 1D signals
            error_tensor_adapted[::, ::, ::self.stride_y] = error_tensor

        else:
            # In case we have a batch of 2D images
            error_tensor_adapted[::, ::, ::self.stride_y, ::self.stride_x] = error_tensor

        # ==============================================================================================================
        # Step 2: Calculate the error tensor for the previous layer
        # ==============================================================================================================
        # Rearrange the weights of the filters to match the dimensionality of the error_images by swaping the number
        # of kernels with the channel size
        filter_kernel_rearranged = np.swapaxes(self.weights, 0, 1)

        # Flip the filter kernels (rotate about 180°)
        filter_kernel_rearranged_and_flipped = filter_kernel_rearranged[::, ::-1]

        # Extract the number of channels
        nr_channels = filter_kernel_rearranged.shape[1]

        # Store all error maps in a batch representation (4D) of 3D error maps
        all_error_maps = None

        # Iterate through the error batch
        for error_image in error_tensor_adapted:
            # Store all convolutions to the one current 1D signal / 2D image in a feature map (2D / 3D numpy.array([]))
            error_map = None

            for filter_kernel in filter_kernel_rearranged_and_flipped:
                # Execute the convolution of the current 1D signal / 2D image with the current kernel
                convolved_image = signal.convolve(error_image, filter_kernel, mode='same')

                # Extract convolution of the center channel
                convolved_image_center_channel = convolved_image[nr_channels // 2]

                # Add the convolved 1D signal / 2D image to a stack to create the error map
                if error_map is None:
                    # Transform to a higher dimensional representation, to be able to stack everything together
                    error_map = np.array([convolved_image_center_channel])

                else:
                    # Add the new convolved 1D signal / 2D image to the stack
                    error_map = np.concatenate((error_map, [convolved_image_center_channel]))

            # Add the created error map to a stack to get a batch representation of all error maps
            if all_error_maps is None:
                # Transform first error map to a batch representation, to be able to stack all error maps together
                all_error_maps = np.array([error_map])

            else:
                # Add the new generated error map to the stack of error maps
                all_error_maps = np.concatenate((all_error_maps, [error_map]))

        # ==============================================================================================================
        # Step 3: Calculate the gradient with respect to the bias.
        # ==============================================================================================================
        # Gradient with respect to the bias is simply sums over the error tensor
        if len(self.convolution_shape) == 2:
            # In case we have a batch of 1D signals
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        else:
            # In case we have a batch of 2D images
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        # ==============================================================================================================
        # Step 4: Pad the input tensor with half the kernels' width. (Necessary for step 5)
        # ==============================================================================================================
        if len(self.convolution_shape) == 2:
            # In case we have a batch of 1D signals, we pad left and right of our data array
            y_pad_left = self.convolution_shape[1] // 2
            y_pad_right = self.convolution_shape[1] // 2

            # Cut the padded image, so the dimensions of the results still match
            if self.convolution_shape[1] % 2 == 0:
                y_pad_right -= 1

            input_tensor_pad = np.pad(self.input_tensor, ((0, 0), (0, 0), (y_pad_left, y_pad_right)),
                                      mode='constant', constant_values=0.0)

        else:
            # In case we have a batch of 2D images we pad around the whole image
            y_pad_left = self.convolution_shape[1] // 2
            y_pad_right = self.convolution_shape[1] // 2
            x_pad_left = self.convolution_shape[2] // 2
            x_pad_right = self.convolution_shape[2] // 2

            # Cut the padded image, so the dimensions of the results still match
            if self.convolution_shape[1] % 2 == 0:
                y_pad_right -= 1
            if self.convolution_shape[2] % 2 == 0:
                x_pad_right -= 1

            input_tensor_pad = np.pad(self.input_tensor, ((0, 0), (0, 0), (y_pad_left, y_pad_right),
                                                          (x_pad_left, x_pad_right)),
                                      mode='constant', constant_values=0.0)

        # ==============================================================================================================
        # Step 5: Calculate the gradient with respect to the weights.
        # ==============================================================================================================

        """
        self.gradient_weights = np.zeros_like(self.weights)

        for c in range(error_tensor.shape[1]):
            kernel_gradient_weights = 0

            for b in range(self.input_tensor.shape[0]):
                inp = input_tensor_pad[b]
                er = error_tensor_adapted[b, c][np.newaxis]
                kernel_gradient_weights += signal.correlate(input_tensor_pad[b], error_tensor_adapted[b, c][np.newaxis], mode='valid')

            self.gradient_weights[c] = kernel_gradient_weights
        """

        # Calculate the kernel as the correlation of input and error channel-wise
        for channel in range(error_tensor.shape[1]):
            kernel = 0

            for input_image_pad, error_image_adopted in zip(input_tensor_pad, error_tensor_adapted[:, channel]):
                # Correlate the padded input images with the adpted error image channel-wise
                # Increase the dimension of error_image_adopted to match the dimensions of input_image_pad
                kernel += signal.correlate(input_image_pad, np.array([error_image_adopted]), mode='valid')

            # Add the calculated kernel to the gradient weights for the respective channel.
            if self.gradient_weights is None:
                # Transform first kernel to a batch representation, to be able to stack all kernels together
                self.gradient_weights = np.array([kernel])

            else:
                # Add the new generated kernel to the stack of kernels
                self.gradient_weights = np.concatenate((self.gradient_weights, [kernel]))

        # Cut the gradient weights onto the same shape as the weights are, so the optimization works.
        self.gradient_weights = self.gradient_weights[:self.weights.shape[0]]

        # ==============================================================================================================
        # Step 6: Optimize the weights.
        # ==============================================================================================================

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        # ==============================================================================================================
        # Step 7: Return the error tensor for the previous layer
        # ==============================================================================================================

        return all_error_maps


    def set_optimizer(self, optimizer):
        """
        Store the optimizer for this layer.
        
        :param optimizer: Optimizer for this layer.
        """

        self.optimizer = optimizer


    def initialize(self, weights_initializer, bias_initializer):
        """
        Reinitialize the weights by using the provided initializer objects.
        
        :param weights_initializer: Initializer object for the weights.
        :param bias_initializer: Initialiser object for the bias.
        """

        # Compute input and output dimension for the weights
        if len(self.convolution_shape) == 2:
            # In case we have a batch of 1D signals
            fan_in = self.weights.shape[1] * self.weights.shape[2]
            fan_out = self.num_kernels * self.weights.shape[2]

        else:
            # In case we have a batch of 2D images
            fan_in = self.weights.shape[1] * self.weights.shape[2] * self.weights.shape[3]
            fan_out = self.num_kernels * self.weights.shape[2] * self.weights.shape[3]

        # Use the respective initializer to reinitialize the weights
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)

        # Use the respective initializer to reinitialize the bias
        bias_shape = self.bias.reshape(self.bias.shape[0], 1, order="C").shape
        self.bias = bias_initializer.initialize(bias_shape, fan_in, fan_out)
