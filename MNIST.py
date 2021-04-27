import numpy as np
import pandas as pd
import math
import random
#This seemed easier than uploading mnist to GitHub.
from keras.datasets import mnist

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# This neural network class has a single hidden layer
class NeuralNetwork:
    def __init__(self, shape):
        self.input = np.zeros(shape[0])
        self.input_bias = np.random.rand(shape[0])
        self.input_weights = np.random.rand(shape[0],shape[1])

        self.hidden_layer = np.zeros(shape[1])
        self.hidden_layer_bias = np.random.rand(shape[1])
        self.hidden_layer_weights = np.random.rand(shape[1],shape[2])

        self.output = np.zeros(shape[2])
        self.output_bias = np.random.rand(shape[2])

        self.cost_function = 0
        self.cost_function_values = np.zeros(len(self.output))

    def getInput(self, datum):
        index = 0
        for i in range(0,datum.shape[0]):
            for j in range(0,datum.shape[1]):
                self.input[index] = datum[i][j]
                index += 1

    def debug(self):
        print(self.input)

    def feedforward(self):

        total = 0

        for j in range(0,len(self.hidden_layer)):
            for i in range(0,len(self.input)):
                total = self.input[i]*self.input_weights[i][j]
            self.hidden_layer[j] = sigmoid(total + self.hidden_layer_bias[j])

        for j in range(0,len(self.output)):
            for i in range(0,len(self.hidden_layer)):
                total = self.output[j]*self.hidden_layer_weights[i][j]
            self.output[j] = sigmoid(total + self.output_bias[j])

    def setCostFunction(self,answer):

        total = 0

        for i in range(len(self.output)):
            if(i != answer):
                total += self.output[i]**2
                self.cost_function_values[i] = self.output[i]**2
            else:
                total+= (self.output[i] - 1)**2
                self.cost_function_values[i] = (self.output[i] - 1)**2

        self.cost_function = total


    def backprop(self):

        hidden_biases = np.zeros(len(self.hidden_layer))

        #Backprop for the weights between output and hidden layer
        for j in range(0,len(self.output)):
            for i in range(0,len(self.hidden_layer)):
                    self.hidden_layer_weights[i][j] = sigmoid(self.cost_function_values[j]) * (1-sigmoid(self.cost_function_values[j]))


        #Backprop for biases in hidden layer
        for i in range(0,len(self.hidden_layer)):
            for j in range(0,len(self.output)):
                hidden_biases[i] = hidden_biases[i] + self.hidden_layer_weights[i][j]

        for i in range(0,len(self.hidden_layer_bias)):
            self.hidden_layer_bias[i] = hidden_biases[i]/28

        for j in range(0,len(self.output)):
                    self.output_bias[j] = sigmoid(self.cost_function_values[j]) * (1-sigmoid(self.cost_function_values[j]))



        #Backprop for biases in output layer
        for i in range(0,len(self.output_bias)):
            self.output_bias[i] = random.random()

        #Backprop for the weights between hidden layer and inputs
        for j in range(0,len(self.hidden_layer)):
            for i in range(0,len(self.input)):
                self.input_weights[i][j] = math.tanh(self.input_weights[i][j] * \
                                          self.hidden_layer_bias[j])





training_set, testing_set = mnist.load_data()
X_train, y_train = training_set
X_test, y_test = testing_set

#Create a neural network with 784 inputs, 10 outputs, and a hidden layer with 16 neurons
MNIST = NeuralNetwork((784,28,10))


#Train the neural network
for i in range(60000):
    MNIST.getInput(X_train[i])
    MNIST.feedforward()
    MNIST.setCostFunction(y_train[i])
    MNIST.backprop()


right = 0
#Test the neural network
for i in range(10000):
    MNIST.getInput(X_test[i])
    MNIST.feedforward()
    MNIST.setCostFunction(y_train[i])
    MNIST.backprop()

    # print(MNIST.output)
    # #print(MNIST.input)
    # print(np.argmax(MNIST.output))
    # print(y_test[i])

    if(np.argmax(MNIST.output) == y_test[i]):
        right += 1

print(right/10000)
