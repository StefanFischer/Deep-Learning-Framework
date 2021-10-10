"""
@description
The Neural Network defines the whole architecture by containing all its layers from the input to the loss layer.
This Network manages the testing and the training.
That means it calls all forward methods passing the data from the beginning to the end,
as well as the optimization by calling all backward passes afterwards.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import copy

# Import parent class
from Optimization.Optimizers import Optimizer


class NeuralNetwork(Optimizer):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        """
        Create the Neural Network.
        
        :param optimizer: Set the used optimization method.
        :param weights_initializer: Set the initializer method for the weights.
        :param bias_initializer: Set the initializer method for the bias.
        """

        # Inherit parent attributes
        super().__init__()

        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        # These members are all set by the UnitTests.
        self.loss = []  # Contain the loss value for each iteration after calling train().
        self.layers = []  # Holds the architecture.
        self.data_layer = None  # Provide input data and labels.
        self.loss_layer = None  # Refer to the special layer providing loss and prediction.

        self.label_tensor = None  # Tensor holds the labels.

        # Define the phase for the layers, either as True for testing, or False for training
        self.phase = None


    @property
    def phase(self):
        """
        Getter for the protected member "phase". 
        """

        return self._phase

    @phase.setter
    def phase(self, phase):
        """
        Setter for the protected member "phase".
        """

        self._phase = phase


    def forward(self):
        """
        Execute the forward pass for the input data through all different layers.
        
        :return: Output of the last layer (i.e. the loss layer) of the network, which is the loss.
        """

        # Access data and save the labels, because they are needed in other functions, too.
        input_tensor, self.label_tensor = self.data_layer.next()

        # Initialize the regularization loss.
        regularization_loss = 0

        # Iterate through all layers till the last (loss) layer.
        for layer in self.layers:
            # Execute forward pass for each layer.
            input_tensor = layer.forward(input_tensor)

            # Calculate the regularization loss for each layer and add them together.
            if self.regularizer:
                regularization_loss += self.regularizer.norm(layer.weights)

        # Calculate the loss by executing the forward pass of the loss layer.
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)

        # Calculate the regularization loss for the loss layer and add it to the overall regularization loss.
        if self.regularizer:
            regularization_loss += self.regularizer.norm(self.loss_layer.weights)

        return loss, regularization_loss

    def backward(self):
        """
        Execute the backward pass by propagating back through the network.
        
        :return: Error tensor of the first layer.
        """

        # Calculate the error tensor of the loss layer
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # Reverse the layer list, because we are back propagating
        reversed_layer_list = reversed(self.layers)

        # Calculate the error tensor of the first layer
        for reversed_layer in reversed_layer_list:
            error_tensor = reversed_layer.backward(error_tensor)

        return error_tensor

    def append_trainable_layer(self, layer):
        """
        Append provided layer to the layers-list.
        
        :param layer: Additional layer, to be added to the layers list.
        """

        # Get a deep copy of the network's optimizer.
        networks_optimizer = copy.deepcopy(self.optimizer)

        # Set the new layer's optimizer to the network's optimizer
        layer.optimizer = networks_optimizer

        # Initialize the layer's weights
        layer.initialize(self.weights_initializer, self.bias_initializer)

        # Append the new layer to the list of layers.
        self.layers.append(layer)

    def train(self, iterations):
        """
        Train the network and store the loss for each iteration.
        
        :param iterations: Number of iterations
        """

        self.phase = False

        # Set the phase of each layer to training
        for layer in self.layers:
            layer.testing_phase = self.phase

        for iteration in range(iterations):
            # Execute the forward pass through the network to obtain the data loss as well as the regularization loss.
            loss, regularization_loss = self.forward()

            # Append the sum of data loss and regularization loss
            self.loss.append(loss + regularization_loss)

            # Execute the backward pass to update the error tensor.
            self.backward()

    def test(self, input_tensor):
        """
        Propagate input_tensor through the network and return the prediction of the last layer.
        For classification tasks we typically query the probabilistic output of the SoftMax layer.
        
        :param input_tensor: Tensor of input data.
        :return: Prediction of the last layer.
        """

        self.phase = True

        for layer in self.layers:
            # Set the phase of each layer to testing
            layer.testing_phase = self.phase

            # Execute forward pass through all layers till the last layer, to get the last layer's prediction.
            input_tensor = layer.forward(input_tensor)

        return input_tensor
