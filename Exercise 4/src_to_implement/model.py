"""
@description
This is the implementation of a variant of the ResNet architecture.
In detail the architecture consists of the following:
    - Conv2D(64, 7, 2)
    - BatchNorm()
    - ReLU()
    - MaxPool(3, 2)
    - ResBlock(64, 1)
        - Sequence of (Conv2D with stride, BatchNorm, ReLU)
        - Sequence of (Conv2D without stride, BatchNorm, ReLU)
        - Batchnorm(1x1Conv(Input)) + Output of upper sequence
    - ResBlock(128, 2)
        - Same as before
    - ResBlock(256, 2)
        - Same as before
    - ResBlock(512, 2)
        - Same as before
    - GlobalAvgPool()
    - Flatten()
    - FC(2)
    - Sigmoid()

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

from torch import nn


class ResNet(nn.Module):
    def __init__(self, in_channels=3, out_channel_dimensions=[64, 64, 128, 256, 512], strides=[2, 1, 2, 2, 2],
                 filter_size=3, activation_func=nn.ReLU(), n_classes=2):
        """
        :param in_channels (int): Channel dimension of the input image (RGB -> 3, Gray -> 1).
        :param out_channel_dimensions ([(int), ...]):  Channel dimensions of the outputs of the convolution operations.
        :param strides ([(int), ...]): Stride sizes for the convolution operations.
        :param filter_size (int): Filter size for the convolution operations.
        :param activation_func (nn.Module): Activation function of the ResNet-blocks.
        :param n_classes (int): Number of class labels.
        """

        # Inherit from parent class
        super().__init__()

        self.encoder = ResNetEncoder(in_channels, out_channel_dimensions, strides, filter_size, activation_func)

        self.decoder = ResnetDecoder(out_channel_dimensions[-1], n_classes)

    def forward(self, input_image):
        """
        Forward Pass for the ResNet

        :param input: Input image.
        :return: Class label.
        """

        # Pass the image through the Encoder's forward pass to obtain the input for the Decoder.
        input_decoder = self.encoder(input_image)

        # Pass the image through the Decoder's forward pass to obtain the class label.
        output = self.decoder(input_decoder)

        return output


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """

    def __init__(self, in_channels, out_channel_dimensions, strides, filter_size, activation_func):
        """
        :param in_channels (int): Channel dimension of the input image (RGB -> 3, Gray -> 1).
        :param out_channel_dimensions ([(int), ...]):  Channel dimensions of the outputs of the convolution operations.
        :param strides ([(int), ...]): Stride sizes for the convolution operations.
        :param filter_size (int): Filter size for the convolution operations.
        :param activation_func (nn.Module): Activation function of the ResNet-blocks.
        """

        # Inherit from parent class
        super().__init__()

        # ==============================================================================================================
        # Step 1: Create the "Gate" of the ResNet Encoder
        #           - Consists of a convolution, a batch normalization, an activation function and a max pooling.
        # ==============================================================================================================
        gate = nn.Sequential(nn.Conv2d(in_channels, out_channel_dimensions[0], kernel_size=filter_size,
                                       stride=strides[0]),
                             nn.BatchNorm2d(num_features=out_channel_dimensions[0]),
                             activation_func,
                             nn.MaxPool2d(kernel_size=filter_size, stride=strides[0]))

        # ==============================================================================================================
        # Step 2: Complete the Encoder by adding four "Res"-blocks with increasing channel dimension.
        # ==============================================================================================================
        res_blocks = nn.Sequential(ResBlock(in_channels=out_channel_dimensions[0],
                                            out_channels=out_channel_dimensions[1], stride=strides[1],
                                            filter_size=filter_size, activation_func=activation_func),

                                   ResBlock(in_channels=out_channel_dimensions[1],
                                            out_channels=out_channel_dimensions[2], stride=strides[2],
                                            filter_size=filter_size, activation_func=activation_func),

                                   ResBlock(in_channels=out_channel_dimensions[2],
                                            out_channels=out_channel_dimensions[3], stride=strides[3],
                                            filter_size=filter_size, activation_func=activation_func),

                                   ResBlock(in_channels=out_channel_dimensions[3],
                                            out_channels=out_channel_dimensions[4], stride=strides[4],
                                            filter_size=filter_size, activation_func=activation_func))

        # ==============================================================================================================
        # Step 3: Store the gate and the res-blocks including their layers and operations in a "Sequential"-object.
        #           -  Layers and operations can now be executed one after the other.
        # ==============================================================================================================
        self.encoder = nn.Sequential(gate, res_blocks)

    def forward(self, input):
        """
        Forward pass of the Encoder.

        :param input: Input data for the Encoder.
        :return: Input for the Decoder.
        """

        # Execute the forward passes of the layers and operations stored in the Encoder in the respective order.
        output = self.encoder(input)

        return output


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of the ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features, nr_classes):
        """
        :param in_features: Number of total features for the Fully Connected Layer.
        :param nr_classes: Number of class labels.
        """

        # Inherit from parent class
        super().__init__()

        # Step 2: Create the Decoder
        self.decoder = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
                                     nn.Flatten(),  # Flatten Layer
                                     nn.Linear(in_features, nr_classes),  # Fully Connected Layer
                                     nn.Sigmoid())  # Sigmoid Activation function

    def forward(self, input):
        """
        Forward pass of the Decoder.

        :param input: Input data for the Decoder.
        :return: Class Label.
        """

        # Execute the forward passes of the layers and operations stored in the Decoder in the respective order.
        output = self.decoder(input)

        return output


class ResBlock(nn.Module):
    """
    Implementation of the ResBlock
    """

    def __init__(self, in_channels, out_channels, stride, filter_size, activation_func):
        """
        :param in_channels (int): Channel dimension of the input.
        :param out_channels (int): Channel dimension of the output.
        :param stride (int): Stride for first Conv2D operation.
        :param filter_size (int): Filter/kernel size for the Conv2D-layers.
        :param activation_func (nn.Module): Activation function.
        """

        # Inherit from parent class
        super().__init__()

        # ==============================================================================================================
        # Step 1: Create the individual parts of the ResBlock.
        # ==============================================================================================================
        # Create the first convolutional operation.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=filter_size,
                               stride=stride, padding=1)

        # Create the second convolutional operation.
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=filter_size, padding=1)

        # Create the batch normalization for the first and second convolutional operation.
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

        # Set the activation function to the provided one.
        self.activation = activation_func

        # ==============================================================================================================
        # Step 2: Create the shortcut around the ResBlock.
        #           - Adjust the size and number of channels if necessary, by applying a 1x1 convolution followed
        #             by a batch normalisation.
        # ==============================================================================================================
        self.shortcut = nn.Sequential()

        # Adjust the size and number of channels if necessary
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                    stride=stride),
                                          nn.BatchNorm2d(num_features=out_channels))

    def forward(self, input_resblock):
        """
        Forward pass of the ResBlock.

        :param input_resblock: Input of the ResBlock.
        :return: Input for next Layer/Block.
        """

        # ==============================================================================================================
        # Step 1: Execute the first convolutional block.
        # ==============================================================================================================
        output_conv_block1 = self.activation(self.batch_norm(self.conv1(input_resblock)))

        # ==============================================================================================================
        # Step 2: Execute the second convolutional block.
        # ==============================================================================================================
        output_conv_block2 = self.batch_norm(self.conv2(output_conv_block1))

        # ==============================================================================================================
        # Step 3: Add the input to the output of the convolutional blocks.
        # ==============================================================================================================
        sum_output_input = output_conv_block2 + self.shortcut(input_resblock)

        # ==============================================================================================================
        # Step 4: Execute the activation function on the result.
        # ==============================================================================================================
        output_resnet_block = self.activation(sum_output_input)

        # ==============================================================================================================
        # Step 5: Return the output of the ResNet-block.
        # ==============================================================================================================
        return output_resnet_block
