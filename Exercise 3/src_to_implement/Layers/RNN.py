"""
@description
The type of recursive neural networks known as Elman network consists of the simplest RNN cells.
They can be modularly implemented as layers.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np

# Import parent class
from Layers.Base import BaseLayer

# Import own written files
from Layers import FullyConnected
from Layers import TanH
from Layers import Sigmoid


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Constructor for the Elman RNN.
        
        :param input_size: Dimension of the input vector. 
        :param hidden_size: Dimension of the hidden state.
        :param output_size: Dimension of the output vector.
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_state = np.zeros(hidden_size)

        # boolean representing whether the RNN regards subsequent sequences as a belonging to the same long sequence.
        self.memorize = False

        # Fully Connected Layers
        self.fc_layer_for_hidden_state = FullyConnected.FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_layer_for_output = FullyConnected.FullyConnected(hidden_size, output_size)

        # Input tensors to the individual time steps "t"
        self.input_tensors_fc_hidden_state = list()
        self.input_tensors_fc_output = list()

        # Activation functions and their results of the forward pass
        self.tanh = TanH.TanH()
        self.tanh_activated_inputs = list()

        self.sigmoid = Sigmoid.Sigmoid()
        self.sigmoid_activated_inputs = list()

        # Gradient weights
        self.gradient_weights = None
        self.gradient_weights_output = None

        # Weights
        self.weights = self.fc_layer_for_hidden_state.weights
        self.weights_output = self.fc_layer_for_output.weights

        # Optimzer
        self.optimizer = None

    @property
    def memorize(self):
        """
        Getter for the protected member "memorize". 
        """

        return self._memorize

    @memorize.setter
    def memorize(self, boolean_state):
        """
        Setter for the protected member "memorize".
        """

        self._memorize = boolean_state

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
    def weights(self):
        """
        Getter for the protected member "weights". 
        """

        return self.fc_layer_for_hidden_state.weights

    @weights.setter
    def weights(self, weights_hidden_state):
        """
        Setter for the protected member "weights".
        """

        self.fc_layer_for_hidden_state.weights = weights_hidden_state

    @property
    def weights_output(self):
        """
        Getter for the protected member "weights_output". 
        """

        return self._weights_output

    @weights_output.setter
    def weights_output(self, weights_output):
        """
        Setter for the protected member "weights_output".
        """

        self._weights_output = weights_output

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


    def calculate_regularization_loss(self):
        """
        Calculate the loss caused by regularization.
        
        :return: Loss caused by regularization.
        """

        if self.optimizer is not None and self.optimizer.regularizer is not None:
            # Calculate the regularization loss for the weights corresponding to the hidden state's FC-Layer.
            regularization_loss_weights_hidden_state = self.optimizer.regularizer.norm(self.weights)

            # Calculate the regularization loss for the weights corresponding to the output's FC-Layer.
            regularization_loss_weights_output = self.optimizer.regularizer.norm(self.weights_output)

            # Calculate the complete regularization loss for the weights corresponding to the FC-Layers.
            regularization_loss = regularization_loss_weights_hidden_state + regularization_loss_weights_output

        else:
            regularization_loss = 0

        return regularization_loss


    def initialize(self, weights_initializer, bias_initializer):
        """
        Initialize the weights and the bias.
        
        :param weights_initializer: Desired initializer for the weights.
        :param bias_initializer: Desired initializer for the bias.
        :return: 
        """

        # Initialize the weights and the bias of the FC-Layer for the hidden state.
        self.fc_layer_for_hidden_state.initialize(weights_initializer, bias_initializer)

        # Initialize the weights and the bias of the FC-Layer for the output.
        self.fc_layer_for_output.initialize(weights_initializer, bias_initializer)


    def forward(self, input_tensor):
        """
        Forward pass for the Elman RNN.
        
        :param input_tensor: Input data of the layer. 
        :return: Input tensor for the next layer.
        """

        # Define the output for all time steps "t"
        output = None

        # ==========================================================================================================
        # Step 1: Get the hidden state of the previous time step "t-1"
        # ==========================================================================================================

        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        for t in range(len(input_tensor)):
            # ==========================================================================================================
            # Step 2: Concatenation of input data with the data of the hidden state and a 1 (bias) to create
            #         x_tilde for the current time step "t".
            #         Do not add the bias here, because in the next step it is added automatically.
            # ==========================================================================================================
            concat_input_hidden = np.array([np.append(input_tensor[t], self.hidden_state)])

            concat_input_hidden_bias = np.concatenate([concat_input_hidden, np.array([np.ones(1)])], axis=1)

            # ==========================================================================================================
            # Step 3: Execute the forward pass of the specified FC-Layer.
            #         This is done to execute the step: "Multiplication of x_tilde with the weight matrix of a FC layer."
            #         Use the concatenation w/o bias, 'cause within the FC-Layer's forward we add the bias automatically.
            # ==========================================================================================================

            dot_product_x_tilde_and_weights_fc = self.fc_layer_for_hidden_state.forward(concat_input_hidden)

            # ==========================================================================================================
            # Step 4: Apply the tanh-activation function to obtain the hidden state of the current time step "t".
            #         Store the new hidden state to use it for the next iteration.
            # ==========================================================================================================

            new_hidden_state = self.tanh.forward(dot_product_x_tilde_and_weights_fc)
            self.hidden_state = new_hidden_state

            # ==========================================================================================================
            # Step 5: Execute the forward pass of the other specified FC-Layer and feed the output to the
            #         sigmoid activation function, to obtain the output for the current time step "t".
            #         Again the bias does not need to be added, since he is added in the forward pass of the FC-layer.
            # ==========================================================================================================

            dot_product_hidden_state_and_weights_fc_output = self.fc_layer_for_output.forward(new_hidden_state)
            output_t = self.sigmoid.forward(dot_product_hidden_state_and_weights_fc_output)

            # ==========================================================================================================
            # Step 6: Add the calculated output of time step "t" to the other outputs of the previous time steps
            # ==========================================================================================================

            if output is None:
                output = output_t

            else:
                output = np.concatenate((output, output_t))

            # ==========================================================================================================
            # Step 7: Store all input_tensors of the FC layers at the time steps "t".
            #         This is necessary for the backward pass of the FC layer, within the RNN backward pass.
            #         The input_tensor at time step "t" needs to be used for the backward pass at the same time step "t".
            # ==========================================================================================================

            self.input_tensors_fc_hidden_state.append(self.fc_layer_for_hidden_state.homogeneous_input_tensor)

            self.input_tensors_fc_output.append(self.fc_layer_for_output.homogeneous_input_tensor)

            # ==========================================================================================================
            # Step 8: Store the results of the forward pass of the tanh and sigmoid activation functions.
            #         This is necessary, so the correct value of the attribute "self.activated_input" are used within
            #         the backward pass.
            # ==========================================================================================================

            self.tanh_activated_inputs.append(new_hidden_state)

            self.sigmoid_activated_inputs.append(output_t)

        # ==============================================================================================================
        # Step 9: Return the output.
        # ==============================================================================================================

        return output

    def backward(self, error_tensor):
        """
        Backward pass for the Elman RNN.
        Update parameters and return error tensor for the previous layer.
        
        :param error_tensor: Error tensor for the Elman RNN.
        :return: Error tensor for the previous layer.
        """

        # ==============================================================================================================
        # Step 1: Reinitialize the gradient weights.
        # ==============================================================================================================

        self.gradient_weights = None
        self.gradient_weights_output = None

        # ==============================================================================================================
        # Step 2: Define variables for the gradient based on the input (will be returned)
        #          and the gradient based on the hidden state (used for the inidividual iterations).
        # ==============================================================================================================

        gradients_input = None
        gradient_hidden_state = np.zeros(self.hidden_size)

        # ==============================================================================================================
        # Step 3: Iterate backwards through the time steps t to do the back propagation.
        # ==============================================================================================================

        for t in reversed(range(len(error_tensor))):
            # ==========================================================================================================
            # Step 4: Load the correct input tensors of the FC layers for the current time step t.
            #          Load the correct "activated_input"-values for both activation functions.
            #          Therefore the backward passes of the individual elements are executed with the correct parameters.
            # ==========================================================================================================

            self.fc_layer_for_hidden_state.homogeneous_input_tensor = self.input_tensors_fc_hidden_state[t]
            self.fc_layer_for_output.homogeneous_input_tensor = self.input_tensors_fc_output[t]

            self.tanh.activated_input = self.tanh_activated_inputs[t]
            self.sigmoid.activated_input = self.sigmoid_activated_inputs[t]

            # ==========================================================================================================
            # Step 5: Backpropagate the gradient of the output through the sigmoid function.
            # ==========================================================================================================

            sigmoid_backward_gradient_output = self.sigmoid.backward(error_tensor[t])

            # ==========================================================================================================
            # Step 6: Backpropagate the gradient of the output through the FC-Layer.
            # ==========================================================================================================

            fc_layer_backward_gradient_output = self.fc_layer_for_output.backward(sigmoid_backward_gradient_output)

            # ==========================================================================================================
            # Step 7: Backpropagate the "Copy"-function of the hidden state as a sum of the gradient of the output
            #        with the gradient of the next hidden state.
            # ==========================================================================================================

            sum_gradients = fc_layer_backward_gradient_output + gradient_hidden_state

            # ==========================================================================================================
            # Step 8: Backpropagate the tanh function.
            # ==========================================================================================================

            tanh_backward_gradient = self.tanh.backward(sum_gradients)

            # ==========================================================================================================
            # Step 9: Backpropagate x_tilde which is the concatenation of the bias,
            #        the input data and the hidden state at the time step "t".
            # ==========================================================================================================

            fc_layer_backward_gradient_x_tilde = self.fc_layer_for_hidden_state.backward(tanh_backward_gradient)

            # ==========================================================================================================
            # Step 10: Split the gradient of x_tilde into the gradient based on the input (will be returned)
            #          and the gradient based on the hidden state (used for the next iteration).
            # ==========================================================================================================

            gradient_input = fc_layer_backward_gradient_x_tilde[:, :self.input_size]
            gradient_hidden_state = fc_layer_backward_gradient_x_tilde[:, self.input_size:]

            # ==========================================================================================================
            # Step 11: Add the calculated gradient based on the input for time step "t" to the gradients
            #          based on the input of the other time steps.
            # ==========================================================================================================

            if gradients_input is None:
                gradients_input = gradient_input

            else:
                gradients_input = np.concatenate((gradients_input, gradient_input))

            # ==========================================================================================================
            # Step 12: Update the gradient with respect to the weights.
            #          .T is necessary, to meet the shape requirements of the Tests.
            # ==========================================================================================================

            if self.gradient_weights is None:
                self.gradient_weights = self.fc_layer_for_hidden_state.gradient_weights
            else:
                self.gradient_weights += self.fc_layer_for_hidden_state.gradient_weights

            if self.gradient_weights_output is None:
                self.gradient_weights_output = self.fc_layer_for_output.gradient_weights
            else:
                self.gradient_weights_output += self.fc_layer_for_output.gradient_weights

        # ==============================================================================================================
        # Step 13: Optimize the weights of the FC Layers, as well as the property weights of this layer.
        # ==============================================================================================================

        if self.optimizer is not None:
            self.fc_layer_for_hidden_state.weights = self.optimizer.calculate_update(self.fc_layer_for_hidden_state.weights, self.gradient_weights)

            self.fc_layer_for_output.weights = self.optimizer.calculate_update(self.fc_layer_for_output.weights, self.gradient_weights_output)

        # ==============================================================================================================
        # Step 14: Reverse the order of gradients based on the input and return the result.
        # ==============================================================================================================

        return gradients_input[::-1]



