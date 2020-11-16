"""
COSC 525 - Deep Learning
Project 2
Contributors:
Metzner, Christoph
Date: 02/10/2020
"""

# Imported Libraries
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

"""
This program describes an artificial neural network (ANN) developed with object-oriented programming using Python 3.
An ANN will consist out of the following three classes:
- Neuron
- FullyConnectedLayer
- NeuralNetwork
Each class represents on its own a distinct level of scale for a typical ANN.
"""


class Neuron:
    def __init__(self, activation_function, number_input, learning_rate, weights=None, bias=None):
        self.activation_function = activation_function
        self.number_input = number_input
        self.learning_rate = learning_rate
        self.bias = bias
        self.updated_bias = None

        if weights is None:
            self.weights = np.random.uniform(0, 1, number_input)
        else:
            self.weights = weights
        # stores output computed within the feed-forward algorithm
        self.output = None
        # stores delta computed within the back-propagation algorithm
        self.delta = None
        # computed updated weights are temporarily stored in an array
        # necessary since back-propagation uses the current weights of neurons in previous layer
        self.updated_weights = []

    # Method for activation of neuron using variable z as input
    # z = bias + sum(weights*inputs)
    # If-statement to select correct activation function based on given string-input ("logistic" or "linear")
    def activate(self, z):
        if self.activation_function == "logistic":
            return log_act(z)
        elif self.activation_function == "linear":
            return lin_act(z)

    # Method for calculating output of neuron based on weighted sum
    # Computing output for neurons of ConvolutionalLayer object requires summation of all elementwise products
    def calculate_conv2d(self, input_vector):
        return self.activate(self.bias + (np.sum(np.multiply(np.array(self.weights), input_vector))))

    # Computing output for neurons of a FullyConnectedLayer object requires np.dot to generate correct results
    def calculate_fully(self, input_vector):
        return self.activate(self.bias + np.dot(self.weights, input_vector))

    # Method to calculate the delta values for the neuron if in the output layer
    def calculate_delta_output(self, actual_output_network, loss_function, number_outputs):
        if loss_function == "mse":
            if self.activation_function == "logistic":
                return mse_loss_prime(self.output, actual_output_network, number_outputs) * log_act_prime(self.output)
            elif self.activation_function == "linear":
                return mse_loss_prime(self.output, actual_output_network, number_outputs) * lin_act_prime(self.output)
        elif loss_function == "bincrossentropy":
            if self.activation_function == "logistic":
                return bin_cross_entropy_loss_prime(self.output, actual_output_network) * log_act_prime(self.output)
            elif self.activation_function == "linear":
                return bin_cross_entropy_loss_prime(self.output, actual_output_network) * lin_act_prime(self.output)

    # Method to calculate the delta values for the neuron if in the hidden layer
    def calculate_delta_hidden(self, delta_sum):
        if self.activation_function == "logistic":
            return delta_sum * log_act_prime(self.output)
        elif self.activation_function == "linear":
            return delta_sum * lin_act_prime(self.output)

    # Method which updates the weight and bias
    # class attributes self.updated_weights and self.updated_bias are cleared or set to None
    # for next training sample iteration
    def update_weights_bias(self):
        self.weights = copy.deepcopy(self.updated_weights)
        self.updated_weights.clear()
        self.bias = copy.deepcopy(self.updated_bias)
        self.updated_bias = None


# Normal fully connected layer
class FullyConnectedLayer:
    def __init__(self, number_neurons, activation_function, number_input, learning_rate, weights=None, bias=None):
        self.number_neurons = number_neurons
        self.activation_function = activation_function
        self.number_input = number_input
        self.learning_rate = learning_rate

        # If bias not given by user create one random bias from a uniform distribution for whole layer
        # this initial bias value is passed on each neuron in respective layer
        if bias is None:
            self.bias = np.random.uniform(0, 1)
        else:
            self.bias = bias
        # self.weights stores all weights for each neuron in the layer
        # those weights are passed down to each respective neuron
        self.weights = weights
        self.neurons = []
        # If no weights given neurons are created without weights (weights are generated in neuron object) and stored in
        # a list; otherwise weights are passed ot respective neuron
        if weights is None:
            for i in range(self.number_neurons):
                self.neurons.append(Neuron(activation_function=activation_function, number_input=self.number_input,
                                           learning_rate=self.learning_rate, weights=None, bias=self.bias))
        else:
            for i in range(self.number_neurons):
                self.neurons.append(Neuron(activation_function=activation_function, number_input=self.number_input,
                                           learning_rate=self.learning_rate, weights=self.weights[i], bias=self.bias))

    # Method calculates the output of each neuron based on the sum of the weights * input + bias of this neuron
    # storing computed output of each neuron in the neuron --> later used for back propagation
    # returns array with final output --> necessary to compute the total accrued loss
    def calculate(self, input_vector):
        output_curr_layer_neuron = []
        for neuron in self.neurons:
            neuron.output = neuron.calculate_fully(input_vector=input_vector)
            output_curr_layer_neuron.append(neuron.output)
        return output_curr_layer_neuron

    # this function calls neuron object method update_weights_bias() to start updating weights for next feed-forward
    # algorithm (For this ANN after each sample --> online processing)
    def update_weights_bias(self):
        for neuron in self.neurons:
            neuron.update_weights_bias()

    def backprop(self, layer_index, actual_output, loss_function, input_layer,
                 deltas_weights=None, input_network=None):
        # reverse the input_vector; actually contains outputs of all layers from feedforward algorithm
        # j = 0 --> output layer
        # j > 0 --> any hidden layer
        if layer_index == 0:
            # Creating a list which contains all deltas of all output neurons
            deltas_weights_output_layer = []
            # Loop: Compute the delta for each neuron in output_neuron
            # actual_output_network[neuron_index]: index of actual output at output neurons of network
            for neuron_index, neuron in enumerate(self.neurons):
                neuron.delta = neuron.calculate_delta_output(actual_output_network=actual_output[neuron_index],
                                                             loss_function=loss_function, number_outputs=1)
                deltas_weights_output_layer.append((neuron.delta, neuron.weights))
                # Computing the gradient / error weight for each weight based on the input to the neuron
                for index_input, current_input in enumerate(input_layer):
                    error_weight = neuron.delta * current_input
                    updated_weight = neuron.weights[index_input] - self.learning_rate * error_weight
                    # updating the weights and bias --> storing the updated weights in class object attributes
                    neuron.updated_weights.append(updated_weight)
                    neuron.updated_bias = neuron.bias - self.learning_rate * neuron.delta
                print()
                print("Current weights: {} \n--> updated weights: \n{}".format(np.round(neuron.weights, 8),
                                                                               np.round(neuron.updated_weights, 8)))
                print("Current bias: {} \n--> updated bias: {}".format(np.round(neuron.bias, 8),
                                                                       np.round(neuron.updated_bias, 8)))
            # compute delta_sum for next layer in backpropagation
            delta_sums = []
            for i in range(len(input_layer)):
                delta_sum = 0
                # delta_sum is used to compute the delta of next layers neurons
                for neuron_index, neuron in enumerate(self.neurons):
                    delta_sum += neuron.delta * neuron.weights[i]
                delta_sums.append(delta_sum)
            return deltas_weights_output_layer, delta_sums

        elif layer_index > 0:
            # Creating a list which contains all deltas of all output neurons
            deltas_weights_output_layer = []
            # Loop: Compute the sum of deltas for each neuron in current layer
            for neuron_index, neuron in enumerate(self.neurons):
                # print("Updated Weights and bias for neuron {} in hidden layer {} seen from output layer:".format(neuron_index + 1, index_layer))
                # setting sum of deltas to 0 for each new neuron
                delta_sum = 0
                # computing delta_sum based on the delta values from neuron in previous layer and the weights
                # connected between those neurons and the neuron in current layer (currently in loop)
                for delta_weight in deltas_weights:
                    delta_sum += delta_weight[0] * delta_weight[1][neuron_index]
                neuron.delta = neuron.calculate_delta_hidden(delta_sum=delta_sum)
                deltas_weights_output_layer.append((neuron.delta, neuron.weights))
                # If statement to check if we reached the end of the network
                # If end is reached then use original network input values as input
                if layer_index == (len(input_layer) - 1):
                    for index_input, current_input in enumerate(input_network):
                        error_weight = neuron.delta * current_input
                        updated_weight = neuron.weights[index_input] - self.learning_rate * error_weight
                        neuron.updated_weights.append(updated_weight)
                # If not reached than continue with the normal input
                else:
                    for index_input, current_input in enumerate(input_layer[layer_index + 1]):
                        error_weight = neuron.delta * current_input
                        updated_weight = neuron.weights[index_input] - self.learning_rate * error_weight
                        neuron.updated_weights.append(updated_weight)
                # updating Bias of neuron
                neuron.updated_bias = neuron.bias - self.learning_rate * neuron.delta
                print()
                print("Current weights: {} \n--> updated weights: \n{}".format(np.round(neuron.weights, 6),
                                                                               np.round(neuron.updated_weights, 6)))
                print("Current bias: {} \n--> updated bias: {}".format(np.round(neuron.bias, 6),
                                                                       np.round(neuron.updated_bias, 6)))
            print()
            # Return the summed deltas for next layer
            # In case the architecture is a NN return deltas and weights
            return deltas_weights_output_layer


class NeuralNetwork:
    def __init__(self, input_size_nn, learning_rate, loss_function):
        self.input_size_nn = input_size_nn
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.NetworkLayers = []
        self.output_each_layer = []
        self.propagated_deltas_weights = None

    # Method to add layers to the Network. Each layer is stored at the end of the list.
    # Thus, order matters.
    def addLayer(self, layer_object):
        self.NetworkLayers.append(layer_object)

    # Method to compute the losses at each output neuron
    # mse_loss: Mean Squared Error
    # bin_cross_entropy_loss: Binary Cross Entropy
    # predicted_output: Output after activation for each output neuron
    # actual_output: Actual output of
    def calculateloss(self, predicted_output, actual_output, number_outputs):
        if self.loss_function == "mse":
            return mse_loss(predicted_output=predicted_output,
                            actual_output=actual_output,
                            number_outputs=number_outputs)
        elif self.loss_function == "bincrossentropy":
            return bin_cross_entropy_loss(predicted_output=predicted_output,
                                          actual_output=actual_output,
                                          number_outputs=number_outputs)

    def update_weights_bias(self):
        # reverse the order of the list containing the individual layer objects
        # necessary for next samples training iteration --> correct feed forward information
        self.NetworkLayers.reverse()
        for layer in self.NetworkLayers:
            layer.update_weights_bias()

    # Feed-Forward algorithm
    def feed_forward(self, current_input_layer):
        print()
        print("Feed-forward Algorithm:")
        print("#######################")
        global output_current_layer
        # For loop through all layers of the network
        # The layers are stored in "self.Networklayers"
        for i, layer in enumerate(self.NetworkLayers):
            print("Layer {}: ".format(layer))
            output_current_layer = layer.calculate(input_vector=current_input_layer)
            # store output per layer in list to use for back-propagation algorithm
            self.output_each_layer.append(output_current_layer)
            current_input_layer = output_current_layer
            print("Output: ", np.round(output_current_layer, 7), sep='\n')
        # Return the final layers output
            print()
        return output_current_layer

    def back_propagation(self, input_network, actual_output):
        print("\n\n")
        print("Back-propagation Algorithm:")
        print("###########################")
        # Reversing list containing all layer objects --> Backpropagation starts at the output layer
        self.NetworkLayers.reverse()  # needed again at end of code for next sample or feed-forward run
        # Reversing list containing each output of all layer objects
        self.output_each_layer.reverse()
        print()
        # For loop to list through network
        for layer_index, layer in enumerate(self.NetworkLayers):
            print("Current Layer: {}".format(layer))
            # Do Back-propagation for output layer
            # In all cases output layer is a FullyConnectedLayer object
            # actual_output: Actual output of network
            if layer_index == 0:
                self.propagated_deltas_weights = layer.backprop(layer_index=layer_index,
                                                                actual_output=actual_output,
                                                                loss_function=self.loss_function,
                                                                # input layer is output of previous layer (FlattenLayer)
                                                                input_layer=self.output_each_layer[layer_index + 1],
                                                                deltas_weights=None,
                                                                input_network=None)
                print()

            # If-statement to handle back-propagation for all intermediate layers
            if layer_index > 0:
                # More if-statements to handle each layer object on its own --> arguments
                if isinstance(layer, FullyConnectedLayer):
                    self.propagated_deltas_weights = layer.backprop(layer_index=layer_index,
                                                                    actual_output=actual_output,
                                                                    loss_function=self.loss_function,
                                                                    # input layer is output of previous layer (FlattenLayer)
                                                                    input_layer=self.output_each_layer,
                                                                    deltas_weights=self.propagated_deltas_weights[0],
                                                                    input_network=input_network)
                    print(self.propagated_deltas_weights)
                    print()
                # Else if statement for the FlattenLayer object
                # returns deltas and weights of previous layer in terms of back-propagation algorithm
                elif isinstance(layer, FlattenLayer):
                    if isinstance(self.NetworkLayers[layer_index+1], MaxPoolingLayer):
                        self.propagated_deltas_weights = layer.backprop(deltas_weights=self.propagated_deltas_weights[1],
                                                                    maxpool=True)
                    else:
                        self.propagated_deltas_weights = layer.backprop(
                            deltas_weights=self.propagated_deltas_weights[1], maxpool=False)
                    print()
                elif isinstance(layer, MaxPoolingLayer):
                    self.propagated_deltas_weights = layer.backprop(deltas_weights=self.propagated_deltas_weights)
                    print()
                elif isinstance(layer, ConvolutionalLayer):
                    # If statement to check whether a maxpoolinglayer was used before the convolutional layer
                    if isinstance(self.NetworkLayers[layer_index - 1], MaxPoolingLayer):
                        self.propagated_deltas_weights = layer.backprop(layer_index=layer_index,
                                                                        delta_sums=self.propagated_deltas_weights[1],
                                                                        input_network=input_network,
                                                                        len_network=len(self.NetworkLayers),
                                                                        input_layer=self.output_each_layer)
                    else:
                        self.propagated_deltas_weights = layer.backprop(layer_index=layer_index,
                                                                        delta_sums=self.propagated_deltas_weights,
                                                                        input_network=input_network,
                                                                        len_network=len(self.NetworkLayers),
                                                                        input_layer=self.output_each_layer)

    # Train Method used to loop through feed_forward and back_propagation algorithm
    # 1 Epoch represents one sample --> online processing
    def train(self, input_network, output_network, epochs=None):
        # Train network based on given argv
        for epoch in range(epochs):
            predicted_output = self.feed_forward(current_input_layer=input_network)
            loss = self.calculateloss(predicted_output=predicted_output,
                                      actual_output=output_network,
                                      number_outputs=1)
            print("Total Loss: ", np.round(np.sum(loss), 5))
            print(np.sum(loss))
            self.back_propagation(input_network=input_network,
                                  actual_output=output_network)


class ConvolutionalLayer:
    def __init__(self, number_kernels, kernel_size, activation_function, dimension_input, learning_rate,
                 bias=None, weights=None, stride=None):
        self.number_kernels = number_kernels  # scalar number of kernels in layer
        self.kernel_size = kernel_size  # vector for kernel size -> 2D, e.g.,  [3,3] = 3x3
        self.activation_function = activation_function  # activation function, e.g., logistic or linear
        self.dimension_input = dimension_input  # dimension of inpt e.g. [2,3]: width=2, height=3
        self.learning_rate = learning_rate  # learning rate is given by NeuralNetwork object

        # default value for stride is set to 1
        if stride is None:
            self.stride = 1
        else:
            self.stride = stride

        # Padding was not included since not required for the task

        # If bias not given by user create one random bias for each kernel of ConvolutionalLayer object
        # from a uniform distribution for whole layer this initial bias value is passed on each
        # neuron in respective layer
        if bias is None:
            self.bias = np.random.uniform(0, 1, self.number_kernels)
        else:
            self.bias = bias

        # generating number of neurons
        neurons_row = ((self.dimension_input[0] - self.kernel_size) // self.stride + 1)  # number of rows with neurons
        neurons_column = ((self.dimension_input[1] - self.kernel_size) // self.stride + 1)  # number of columns with neurons
        self.number_neurons = neurons_row * neurons_column * self.number_kernels  # number of total neurons in layer

        # List holding all neurons in layer
        self.feature_maps_layer = []
        # Loop through all kernels to generate neurons
        for kernel in range(number_kernels):
            # list holding all neuron objects of one kernel
            self.feature_map = []
            # if-statement to check whether weights were given or not...
            # if not given, then generate weights for the kernel_size (quadratic) which all neurons share
            if weights is None:
                self.weights = []
                # for loop to generate weights in 2D-matrix
                for row in range(self.kernel_size):
                    self.weights.append(np.random.uniform(0, 1, self.kernel_size))
            else:
                self.weights = weights
            # generating neurons of feature maps, with respective weights and one bias
            for row in range(neurons_row):
                self.neurons = []
                for column in range(neurons_column):
                    self.neurons.append(
                        Neuron(activation_function=self.activation_function, number_input=self.kernel_size,
                               learning_rate=self.learning_rate, weights=self.weights[kernel], bias=self.bias[kernel]))
                self.feature_map.append(self.neurons)
            self.feature_maps_layer.append(self.feature_map)

    def calculate(self, input_vector):
        # List which contains all output for each feature map
        output_feature_maps = []
        # First for-loop: Loop through all feature maps
        for feature_map in self.feature_maps_layer:
            # Starting location of kernel is index (0, 0) in matrix
            # Defines row position/index of upper end of kernel
            kern_edge_top = 0
            # Defines row position/index of lower end of kernel
            kern_edge_bot = self.kernel_size
            output_feature_map = []
            for neuron_row in feature_map:
                # Defines col position/index of left side of kernel
                kern_edge_l = 0
                # Defines col position/index of right side of kernel
                kern_edge_r = self.kernel_size
                # Empty list to contain all neurons in one row
                output_neurons_row = []
                for neuron in neuron_row:
                    # If statement to check whether input has same size as kernel
                    # no moving of kernel needed to compute the output of the neuron
                    if self.dimension_input[0] == self.kernel_size:
                        print(input_vector)
                        neuron.output = neuron.calculate_conv2d(input_vector=input_vector)
                    else:
                        # generating a snippet of input matrix based on size of kernel and location
                        reshaped_input_vector = input_vector[kern_edge_top:kern_edge_bot, kern_edge_l:kern_edge_r]
                        neuron.output = neuron.calculate_conv2d(input_vector=reshaped_input_vector)
                    # Updating the location of kernel for next neuron via stride - left and right side of kernel
                    kern_edge_l += self.stride
                    kern_edge_r += self.stride
                    # Generating output matrix
                    output_neurons_row.append(neuron.output)
                output_feature_map.append(output_neurons_row)
                # Updating the location of kernel for neuron via stride - top and bottom side of kernel
                kern_edge_top += self.stride
                kern_edge_bot += self.stride
            output_feature_maps.append(output_feature_map)
        return output_feature_maps

    def backprop(self, layer_index, delta_sums, input_network, len_network, input_layer=None):
        # First Step: Compute deltas for each neuron, where required --> maxpooling
        # If previous layer was no MaxPoolingLayer object, than ignor the following step
        # generating number of neurons
        delta_sums_kernels = []
        for feature_map_index, feature_map in enumerate(self.feature_maps_layer):
            print("Current Feature Map: ", feature_map_index+1)
            if layer_index == (len(input_layer) - 1):
                pass
            else:
                input_layer = np.array(input_layer[layer_index + 1][feature_map_index])
            # We need to generate the gradients for each neuron

            sum_gradient = np.zeros((self.kernel_size, self.kernel_size))
            # Nested for-loop to get access to each neuron
            bias_gradient = 0
            for row_neuron, row_feature_map in enumerate(feature_map):
                for col_neuron, neuron in enumerate(row_feature_map):
                    # print("Neuron {} at Row / Col ({}, {}).".format(neuron, row_neuron, col_neuron))
                    # print("Delta sum:", delta_sums[feature_map_index][row_neuron][col_neuron])
                    neuron.delta = neuron.calculate_delta_hidden(
                        delta_sums[feature_map_index][row_neuron][col_neuron])
                    # Sum over deltas of all neurons in feature map to get gradient of bias
                    bias_gradient += neuron.delta

                    # Reshaping input matrix into input w.r.t current neuron
                    # if statement to give correct input in layer
                    # if end of network reached provide input in network
                    if layer_index == (len_network-1):
                        current_input = np.array(input_network[row_neuron:self.kernel_size + row_neuron,
                                                 col_neuron:self.kernel_size + col_neuron])
                    else:
                        current_input = np.array(input_layer[row_neuron:self.kernel_size + row_neuron,
                                                 col_neuron:self.kernel_size + col_neuron])
                    # add for loop which walks down the single input areas with weights / gradients
                    # This is the convolution!
                    # print(current_input)
                    # print(neuron.delta)
                    sum_gradient += np.multiply(current_input, neuron.delta)
                    # print(sum_gradient)
            print("Current Weights: \n", self.weights[feature_map_index])
            # generating updated_weights and biases per kernel index
            updated_weight = self.weights[feature_map_index] - np.multiply(self.learning_rate, sum_gradient)
            print("Updated Weights: \n", updated_weight)
            updated_bias = self.bias[feature_map_index] - self.learning_rate * bias_gradient
            print("Current Bias: ", self.bias[feature_map_index])
            print("Updated Bias: ", updated_bias)
            # Doing "convolution" here!
            delta_sums_kernels.append(sum_gradient*self.weights[feature_map_index])
            # update weights for each neuron in this feature_map
            for row_feature_map in feature_map:
                for neuron in row_feature_map:
                    neuron.updated_weights.append(updated_weight)
                    neuron.updated_bias = updated_bias
        # return deltas for next layer in back prop
            print()
        return delta_sums_kernels


class MaxPoolingLayer:
    def __init__(self, kernel_size, dimension_input):
        self.kernel_size = kernel_size
        self.dimension_input = dimension_input
        self.stride = kernel_size
        self.maxpool_feature_maps = []
        self.maxpool_index_masks = []

    def calculate(self, input_vector):
        input_vector = np.array(input_vector)
        # Setting up certain variables necessary for maxpooling
        number_input_kernels = len(input_vector)
        number_strides_rows = self.dimension_input[0] // self.stride
        number_strides_cols = self.dimension_input[1] // self.stride
        # for loop through all kernel_index
        for kernel_index in range(number_input_kernels):
            # Starting location of kernel is index (0, 0) in matrix
            # Defines row position/index of upper end of kernel
            kern_edge_top = 0
            # Defines row position/index of lower end of kernel
            kern_edge_bot = self.kernel_size
            maxpool_feature_map = []
            maxpool_index_mask = []
            for stride_index_row in range(number_strides_rows):
                # Defines column position/index of left side of kernel
                kern_edge_l = 0
                # Defines column position/index of right side of kernel
                kern_edge_r = self.kernel_size
                maxpool_index_row = []
                maxpool_row = []
                for stride_index_col in range(number_strides_cols):
                    # Set matrix for current maxpool kernel from feature map
                    current_maxpool = input_vector[kernel_index][kern_edge_top:kern_edge_bot, kern_edge_l:kern_edge_r]
                    # find max value in current maxpool matrix
                    maxpool_value = np.max(current_maxpool)
                    # Get index of max value in current maxpool matrix
                    maxpool_index = np.where(current_maxpool == maxpool_value)
                    # Set current index in context of whole input matrix
                    maxpool_index_input = [stride_index_row*self.stride+maxpool_index[0],
                                           stride_index_col*self.stride+maxpool_index[1]]
                    maxpool_index_row.append((maxpool_value, maxpool_index_input))
                    maxpool_row.append(maxpool_value)
                    # Updating the location of kernel for next neuron via stride - left and right side of kernel
                    kern_edge_l += self.stride
                    kern_edge_r += self.stride
                maxpool_index_mask.append(maxpool_index_row)
                maxpool_feature_map.append(maxpool_row)
                # Updating the location of kernel for neuron via stride - top and bottom side of kernel
                kern_edge_top += self.stride
                kern_edge_bot += self.stride
            # Storing index locations of all max values in list
            self.maxpool_index_masks.append(maxpool_index_mask)
            # Storing all max values in list which is forwarded to next layer as the input
            # In our case its the flatten layer
            self.maxpool_feature_maps.append(maxpool_feature_map)
        return self.maxpool_feature_maps

    def backprop(self, deltas_weights):
        # List containing masks with all index locations and value of maxpooling process
        maxpool_masks = []
        for i, maxpool_index_mask in enumerate(self.maxpool_index_masks):
            maxpool_mask = np.zeros(self.dimension_input)
        # Loop through each maxpool_index_mask --> locations of max values
            for j, maxpool_index_mask_row in enumerate(maxpool_index_mask):
                for k, maxpool in enumerate(maxpool_index_mask_row):
                    # set gradient from previous layer equal to location of max value of pooling layer (feed forward)
                    row_index = maxpool[1][0][0]
                    col_index = maxpool[1][1][0]
                    maxpool_mask[row_index][col_index] = deltas_weights[i][j][k]
            maxpool_masks.append(maxpool_mask)
            # two variables are returned, situational if maxpool is included network or not
        return deltas_weights, maxpool_masks


class FlattenLayer:
    def __init__(self, dimension_input):
        self.dimension_input = dimension_input

    def calculate(self, input_vector):
        # Example format of input_vector is as following: input_vector = [[[a,b,c,d], [w,x,y,z]]]
        # Flatten list to achieve same format as Keras
        # Keras flattens the list as follows
        # list_A = [a,b,c,d]
        # list_B = [w,x,y,z]
        # FlattenLayer = [a,w,b,x,c,y,d,z]
        input_flattened_lists = []
        for list_element in input_vector:
            single_list = []
            for row in list_element:
                for col in row:
                    single_list.append(col)
            input_flattened_lists.append(single_list)

        flatten_layer = []
        for i in range(len(input_flattened_lists[0])):
            for list_element in input_flattened_lists:
                flatten_layer.append(list_element[i])
        # np.array(input_vector).flatten()
        return flatten_layer

    def backprop(self, deltas_weights, maxpool):
        # If statement to check if previous layer was a MaxPoolingLayer object
        # Code is hardcoded for example 3
        # Not a generalization of grouping a list, like List = [a,b,a,b,a,b,a,b]
        # Final list has format: List = [a,a,a,a,b,b,b,b] where the order matters for each distinct group element
        if maxpool is True:
            b = []
            c = []
            for index, element in enumerate(deltas_weights):
                if index % 2 == 0:
                    b.append(element)
                else:
                    c.append(element)
            #
            deltas_weights.clear()
            deltas_weights = b + c
        elif maxpool is False:
            pass
        # reshape input deltas_weights into certain shape depending on the dimension of input
        delta_sum = np.reshape(deltas_weights, (len(deltas_weights)//np.product(self.dimension_input),
                                                self.dimension_input[0], self.dimension_input[1]))
        return delta_sum


"""
Activation Functions with their respective prime functions
- logistic (log_act)
- linear (lin_act)
- ReLU (ReLU_act)
- z: result of the weighted sum of weights (w), biases (b), and inputs (x) - z = np.dot(w,x)-b
"""


# sigmoid/logistic activation function
def log_act(z):
    return 1 / (1 + np.exp(-z))


def log_act_prime(output):
    return output * (1 - output)


# linear activation function
def lin_act(z):
    return z


def lin_act_prime(z):
    return 1


"""
Loss Functions
- Mean squared error (mse_loss); https://en.wikipedia.org/wiki/Mean_squared_error
- Binary cross entropy loss; https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html (bin_cross_entropy_loss)
- predicted: Array containing all predicted/computed output for each sample by the neural network.
- actual: Array containing the ground truth value for each sample, respectively.
"""


def mse_loss(predicted_output, actual_output, number_outputs):
    output_network = []
    for output_index, output in enumerate(predicted_output):
        loss = 1 / number_outputs * (actual_output[output_index] - output) ** 2
        output_network.append(loss)
    return output_network


def mse_loss_prime(predicted_output, actual_output, number_outputs):
    return 2 * (actual_output - predicted_output) * (1/number_outputs) * -(1)


def bin_cross_entropy_loss(predicted_output, actual_output, number_outputs):
    return 1 / number_outputs * -(actual_output * np.log(predicted_output)
                                  + (1 - actual_output) * np.log(1 - predicted_output))


def bin_cross_entropy_loss_prime(predicted_output, actual_output):
    return -(predicted_output / actual_output) + ((1 - predicted_output) / (1 - actual_output))


# Driver code main()
def main(argv=None):
    # First List: List holding all weights for each kernel
    # Second List: Holds weights for one kernel
    # Third List: Holds weights for first row of kernel weight 00, 01, 02 / 10, 11, 12 / 20, 21, 22
    # print(NN.NetworkLayers[layer_index].neurons_layer[kernel_index_neurons][row_index_neurons][column_index_neurons])
    # cov_res = [4,3,4], [2,4,3], [2, 3, 4]
    if argv[1] == 'example1':
        input_example1 = [[0.02, 0.21, 0.07, 0.17, 0.78],
                          [0.09, 0.25, 0.78, 0.04, 0.24],
                          [0.97, 0.29, 0.37, 0.27, 0.82],
                          [1.00, 0.29, 0.75, 0.62, 0.56],
                          [0.88, 0.65, 0.09, 0.99, 0.87]]
        weights_conv2d = [[[0.12, 0.89, 0.21], [0.04, 0.64, 0.13], [0.91, 0.05, 0.64]]]
        bias_conv2D = [1]

        weights_dense = [[0.99, 0.93, 0.83, 0.49, 0.59, 0.3, 0.96, 0.72, 0.8]]
        bias_dense = [1.5]

        output_example1 = 0.5

        NN = NeuralNetwork(input_size_nn=2, learning_rate=0.5, loss_function="mse")
        NN.addLayer(ConvolutionalLayer(number_kernels=1, kernel_size=3, activation_function="logistic",
                                       dimension_input=[5, 5], learning_rate=0.5, bias=bias_conv2D,
                                       weights=weights_conv2d,
                                       stride=None))
        NN.addLayer(FlattenLayer(dimension_input=[3, 3]))
        NN.addLayer(FullyConnectedLayer(number_neurons=1, activation_function="logistic",
                                        number_input=np.product([3, 3]), learning_rate=0.5, weights=weights_dense,
                                        bias=bias_dense))
        NN.train(input_network=np.array(input_example1),
                 output_network=[output_example1],
                 epochs=1)

    elif argv[1] == 'example2':
        input_example2 = [[0.02, 0.21, 0.07, 0.17, 0.78],
                          [0.09, 0.25, 0.78, 0.04, 0.24],
                          [0.97, 0.29, 0.37, 0.27, 0.82],
                          [1.00, 0.29, 0.75, 0.62, 0.56],
                          [0.88, 0.65, 0.09, 0.99, 0.87]]
        weights_example2_conv1 = [[[0.12, 0.89, 0.21], [0.04, 0.64, 0.13], [0.91, 0.05, 0.64]]]
        weights_example2_conv2 = [[[0.49, 0.98, 0.89], [0.46, 0.47, 0.44], [0.26, 0.65, 0.87]]]
        output_example2 = 0.5

        weights_full = [[0.5]]
        bias_full = [1.5]

        NN = NeuralNetwork(input_size_nn=2, learning_rate=0.5, loss_function="mse")
        NN.addLayer(ConvolutionalLayer(number_kernels=1, kernel_size=3, activation_function="logistic",
                                       dimension_input=[5, 5], learning_rate=0.5, bias=[2],
                                       weights=weights_example2_conv1, stride=None))
        NN.addLayer(ConvolutionalLayer(number_kernels=1, kernel_size=3, activation_function="logistic",
                                       dimension_input=[3, 3], learning_rate=0.5, bias=[2],
                                       weights=weights_example2_conv2, stride=None))
        NN.addLayer(FlattenLayer(dimension_input=[1, 1]))
        NN.addLayer(FullyConnectedLayer(number_neurons=1, activation_function="logistic",
                                        number_input=np.product([1]), learning_rate=0.5, weights=weights_full,
                                        bias=bias_full))
        NN.train(input_network=np.array(input_example2),
                 output_network=[output_example2],
                 epochs=1)

    elif argv[1] == 'example3':
        input_example3 = [[0.61, 0.73, 0.42, 0.97, 0.77, 0.68],
                          [0.80, 0.06, 0.39, 0.11, 0.10, 0.65],
                          [0.95, 0.95, 0.70, 0.57, 0.47, 0.98],
                          [0.32, 0.08, 0.69, 0.02, 0.89, 0.07],
                          [0.58, 0.31, 0.21, 0.03, 0.04, 0.04],
                          [0.93, 0.67, 0.84, 0.7, 0.36, 0.08]]

        weights_example3 = [[[0.12, 0.89, 0.21], [0.04, 0.64, 0.13], [0.91, 0.64, 0.65]],
                            [[0.49, 0.98, 0.89], [0.46, 0.47, 0.44], [0.05, 0.26, 0.87]]]

        bias_example3 = [0.95, 1.0]

        weights_full = [[0.99, 0.93, 0.83, 0.49, 0.59, 0.3, 0.96, 0.72]]
        bias_full = [1.5]
        output_example3 = 0.5
        NN = NeuralNetwork(input_size_nn=2, learning_rate=0.5, loss_function="mse")
        NN.addLayer(ConvolutionalLayer(number_kernels=2, kernel_size=3, activation_function="logistic",
                                       dimension_input=[6, 6], learning_rate=0.5, bias=bias_example3,
                                       weights=weights_example3,
                                       stride=1))
        NN.addLayer(MaxPoolingLayer(kernel_size=2, dimension_input=[4, 4]))
        NN.addLayer(FlattenLayer(dimension_input=[2, 2]))
        NN.addLayer(FullyConnectedLayer(number_neurons=1, activation_function="logistic",
                                        number_input=8, learning_rate=0.5, weights=weights_full, bias=bias_full))
        NN.train(input_network=np.array(input_example3),
                 output_network=[output_example3],
                 epochs=1)

    elif argv[1] == "project1":
        input_network = [0.05, 0.10]
        weights_hidden = [[0.15, 0.20], [0.25, 0.30]]
        bias_hidden = [0.35]
        weights_output = [[0.40, 0.45], [0.50, 0.55]]
        bias_output = [0.60]

        output_network = [0.01, 0.99]

        NN = NeuralNetwork(input_size_nn=[2, 1], learning_rate=0.5, loss_function="mse")
        # Hidden_layer
        NN.addLayer(FullyConnectedLayer(number_neurons=2, activation_function="logistic",
                                        number_input=np.product([2, 1]), learning_rate=0.5,
                                        weights=weights_hidden, bias=bias_hidden))
        # Output_layer
        NN.addLayer(FullyConnectedLayer(number_neurons=2, activation_function="logistic",
                                        number_input=np.product([2, 1]), learning_rate=0.5,
                                        weights=weights_output, bias=bias_output))
        # NN.print_network()
        # starting the training algorithm
        NN.train(input_network=input_network, output_network=output_network, epochs=1)


if __name__ == '__main__':
    main(sys.argv)
